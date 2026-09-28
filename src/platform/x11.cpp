// ============================================================================
//  platform/x11.cpp — ventana X11 con presentación MIT-SHM de doble búfer.
//
//  * Dos XImage en memoria compartida (shmget/shmat + XShmAttach). present()
//    escala el framebuffer (AVX2, paralelo) dentro de la imagen libre y llama a
//    XShmPutImage(send_event=True); el servidor avisa con ShmCompletion cuando
//    terminó de leerla. Nunca escribimos una imagen que el servidor aún lee: si
//    las dos están ocupadas se espera a su ShmCompletion (XCheckTypedEvent +
//    poll() sobre el socket, sin consumir los eventos de entrada).
//  * El segmento se marca IPC_RMID nada más adjuntarlo: si el proceso muere, el
//    kernel lo libera (no quedan segmentos huérfanos).
//  * Sin MIT-SHM (servidor remoto) → XPutImage clásico por el socket.
//  * Entrada: XLookupString → texto Latin-1 (con composición propia de teclas
//    muertas ´ ¨ ~ ` ^ para teclados españoles), keysym sin modificadores →
//    Key, autorepetición "detectable" de XKB (sin KeyRelease falsos), ruedas
//    4/5, doble clic, deltas de ratón con resto (sin perder sub-píxeles al /scale).
// ============================================================================
#include "platform.hpp"
#include "../core/util.hpp"

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <X11/keysym.h>
#include <poll.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <cstdlib>
#include <cstring>

namespace cfd::platform {
namespace {

bool g_x_error = false;
int x_error_trap(Display*, XErrorEvent*) { g_x_error = true; return 0; }

CFD_INLINE int floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

int map_keysym(KeySym ks) {
    switch (ks) {
        case XK_Escape: return KeyEscape;
        case XK_Return: case XK_KP_Enter: return KeyEnter;
        case XK_Tab: case XK_ISO_Left_Tab: return KeyTab;
        case XK_BackSpace: return KeyBackspace;
        case XK_Delete: case XK_KP_Delete: return KeyDelete;
        case XK_Insert: case XK_KP_Insert: return KeyInsert;
        case XK_Left: case XK_KP_Left: return KeyLeft;
        case XK_Right: case XK_KP_Right: return KeyRight;
        case XK_Up: case XK_KP_Up: return KeyUp;
        case XK_Down: case XK_KP_Down: return KeyDown;
        case XK_Home: case XK_KP_Home: return KeyHome;
        case XK_End: case XK_KP_End: return KeyEnd;
        case XK_Prior: case XK_KP_Prior: return KeyPageUp;
        case XK_Next: case XK_KP_Next: return KeyPageDown;
        case XK_Shift_L: case XK_Shift_R: return KeyShift;
        case XK_Control_L: case XK_Control_R: return KeyCtrl;
        case XK_Alt_L: case XK_Alt_R: case XK_Meta_L: case XK_Meta_R: return KeyAlt;
        case XK_KP_Add: return '+';
        case XK_KP_Subtract: return '-';
        case XK_KP_Multiply: return '*';
        case XK_KP_Divide: return '/';
        case XK_KP_Decimal: case XK_KP_Separator: return '.';
        default: break;
    }
    if (ks >= XK_F1 && ks <= XK_F12) return KeyF1 + static_cast<int>(ks - XK_F1);
    if (ks >= XK_KP_0 && ks <= XK_KP_9) return '0' + static_cast<int>(ks - XK_KP_0);
    if (ks >= 0x20 && ks <= 0x7E) return (ks >= 'A' && ks <= 'Z') ? static_cast<int>(ks + 32) : static_cast<int>(ks);
    // Letras Latin-1 (ñ, ç...): keysym = código Latin-1 (extensión: 160..255 < KeyEscape).
    if (ks >= 0xA0 && ks <= 0xFF) return static_cast<int>(ks);
    return KeyNone;
}

// Composición de teclas muertas → Latin-1 (0 si no existe).
u8 compose_dead(KeySym dead, u8 ch) {
    static constexpr const char* base = "aeiouAEIOUnNcCyY ";
    // columnas: agudo, grave, circunflejo, diéresis, tilde
    static constexpr u8 tab[17][5] = {
        {0xE1, 0xE0, 0xE2, 0xE4, 0xE3}, {0xE9, 0xE8, 0xEA, 0xEB, 0}, {0xED, 0xEC, 0xEE, 0xEF, 0},
        {0xF3, 0xF2, 0xF4, 0xF6, 0xF5}, {0xFA, 0xF9, 0xFB, 0xFC, 0}, {0xC1, 0xC0, 0xC2, 0xC4, 0xC3},
        {0xC9, 0xC8, 0xCA, 0xCB, 0},    {0xCD, 0xCC, 0xCE, 0xCF, 0},    {0xD3, 0xD2, 0xD4, 0xD6, 0xD5},
        {0xDA, 0xD9, 0xDB, 0xDC, 0},    {0, 0, 0, 0, 0xF1},             {0, 0, 0, 0, 0xD1},
        {0xE7, 0, 0, 0, 0},             {0xC7, 0, 0, 0, 0},             {0xFD, 0, 0, 0xFF, 0},
        {0xDD, 0, 0, 0, 0},             {0xB4, '`', '^', 0xA8, '~'},
    };
    int col;
    switch (dead) {
        case XK_dead_acute: col = 0; break;
        case XK_dead_grave: col = 1; break;
        case XK_dead_circumflex: col = 2; break;
        case XK_dead_diaeresis: col = 3; break;
        case XK_dead_tilde: col = 4; break;
        default: return 0;
    }
    for (int i = 0; base[i]; ++i)
        if (static_cast<u8>(base[i]) == ch) return tab[i][col];
    return 0;
}

class X11Window final : public Window {
public:
    explicit X11Window(Display* d) : dpy_(d) {}
    ~X11Window() override { close(); }

    bool open(const char* title, int fb_w, int fb_h, int pixel_scale) override;
    void poll(Input& in) override;
    void present(const render::Framebuffer& fb) override;
    void set_title(const char* title) override;
    int pixel_scale() const override { return scale_; }
    bool is_headless() const override { return false; }
    PresentStats present_stats() const override { return stats_; }
    void set_fullscreen(bool on) override;
    bool fullscreen() const override { return fullscreen_; }

private:
    struct Img {
        XImage* img = nullptr;
        XShmSegmentInfo seg{};
        u32* data = nullptr;
        int w = 0, h = 0, stride = 0;   // stride en píxeles
        bool busy = false;
        bool shm = false;
    };
    bool create_images(int w, int h);
    void destroy_images();
    bool create_image(Img& im, int w, int h, bool try_shm);
    void wait_image(Img& im, double timeout_s);
    void on_completion(const XEvent& ev);
    void handle(XEvent& ev, Input& in);
    void release_all(Input& in);
    void close();

    Display* dpy_ = nullptr;
    ::Window win_ = 0;
    int screen_ = 0;
    Visual* vis_ = nullptr;
    int depth_ = 24;
    GC gc_ = nullptr;
    Atom wm_protocols_ = 0, wm_delete_ = 0, net_wm_name_ = 0, utf8_ = 0, net_state_ = 0, net_fs_ = 0;
    int scale_ = 1;
    int win_w_ = 0, win_h_ = 0;
    int fb_w_ = 0, fb_h_ = 0;
    bool size_changed_ = false;
    bool shm_ok_ = false;
    int shm_event_ = -1;
    Img img_[2];
    int cur_ = 0;
    bool fullscreen_ = false;
    bool nt_ = true;                          // stores no temporales en el escalado (CFD_NO_NT=1 los quita)
    // Estado de entrada entre cuadros
    int last_wx_ = 0, last_wy_ = 0;           // última posición (píxeles de ventana)
    int acc_dx_ = 0, acc_dy_ = 0;             // deltas acumulados (píxeles de ventana)
    bool pointer_valid_ = false;
    Time last_click_ = 0;
    int last_click_x_ = -1000, last_click_y_ = -1000;
    KeySym dead_ = 0;
    PresentStats stats_;
};

bool X11Window::open(const char* title, int fb_w, int fb_h, int pixel_scale) {
    if (!dpy_) return false;
    scale_ = clamp_(pixel_scale, 1, 8);
    fb_w_ = max_(fb_w, 1); fb_h_ = max_(fb_h, 1);
    win_w_ = fb_w_ * scale_; win_h_ = fb_h_ * scale_;
    screen_ = DefaultScreen(dpy_);
    vis_ = DefaultVisual(dpy_, screen_);
    depth_ = DefaultDepth(dpy_, screen_);
    if (vis_->c_class != TrueColor || (depth_ != 24 && depth_ != 32) || vis_->red_mask != 0xFF0000 ||
        vis_->green_mask != 0xFF00 || vis_->blue_mask != 0xFF) {
        std::fprintf(stderr, "[x11] visual no soportado (se necesita TrueColor 24/32 bits BGRA)\n");
        return false;
    }
    const ::Window root = RootWindow(dpy_, screen_);
    XSetWindowAttributes a{};
    a.background_pixmap = None;              // sin borrado del servidor → sin parpadeo al redimensionar
    a.bit_gravity = NorthWestGravity;
    a.event_mask = KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                   StructureNotifyMask | FocusChangeMask | ExposureMask | LeaveWindowMask | EnterWindowMask;
    win_ = XCreateWindow(dpy_, root, 0, 0, static_cast<unsigned>(win_w_), static_cast<unsigned>(win_h_), 0, depth_,
                         InputOutput, vis_, CWBackPixmap | CWBitGravity | CWEventMask, &a);
    if (!win_) return false;
    wm_protocols_ = XInternAtom(dpy_, "WM_PROTOCOLS", False);
    wm_delete_ = XInternAtom(dpy_, "WM_DELETE_WINDOW", False);
    net_wm_name_ = XInternAtom(dpy_, "_NET_WM_NAME", False);
    utf8_ = XInternAtom(dpy_, "UTF8_STRING", False);
    net_state_ = XInternAtom(dpy_, "_NET_WM_STATE", False);
    net_fs_ = XInternAtom(dpy_, "_NET_WM_STATE_FULLSCREEN", False);
    XSetWMProtocols(dpy_, win_, &wm_delete_, 1);
    XClassHint ch{};
    char res_name[] = "cfd", res_class[] = "CFD";
    ch.res_name = res_name; ch.res_class = res_class;
    XSetClassHint(dpy_, win_, &ch);
    XSizeHints* sh = XAllocSizeHints();
    if (sh) {
        sh->flags = PMinSize;
        sh->min_width = 320 * scale_ / 2; sh->min_height = 200 * scale_ / 2;
        XSetWMNormalHints(dpy_, win_, sh);
        XFree(sh);
    }
    set_title(title);
    XGCValues gv{};
    gv.graphics_exposures = False;
    gc_ = XCreateGC(dpy_, win_, GCGraphicsExposures, &gv);
    Bool supported = False;
    XkbSetDetectableAutoRepeat(dpy_, True, &supported);
    // MIT-SHM
    int major = 0, minor = 0;
    Bool pixmaps = False;
    shm_ok_ = XShmQueryExtension(dpy_) && XShmQueryVersion(dpy_, &major, &minor, &pixmaps);
    if (shm_ok_) shm_event_ = XShmGetEventBase(dpy_) + ShmCompletion;
    if (std::getenv("CFD_NO_SHM")) shm_ok_ = false;
    nt_ = std::getenv("CFD_NO_NT") == nullptr;
    XMapWindow(dpy_, win_);
    XFlush(dpy_);
    if (!create_images(win_w_, win_h_)) return false;
    size_changed_ = true;                     // el primer poll informa del tamaño
    return true;
}

void X11Window::set_title(const char* title) {
    if (!dpy_ || !win_) return;
    const char* t = title ? title : "";
    XChangeProperty(dpy_, win_, net_wm_name_, utf8_, 8, PropModeReplace, reinterpret_cast<const unsigned char*>(t),
                    static_cast<int>(std::strlen(t)));
    // WM_NAME en Latin-1 (gestores antiguos): conversión UTF-8 → Latin-1 sencilla.
    char l1[256];
    int n = 0;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(t); *p && n < 255;) {
        if (*p < 0x80) { l1[n++] = static_cast<char>(*p++); }
        else if ((*p & 0xE0) == 0xC0 && (p[1] & 0xC0) == 0x80) {
            const u32 cp = ((p[0] & 0x1Fu) << 6) | (p[1] & 0x3Fu);
            l1[n++] = static_cast<char>(cp <= 0xFF ? cp : '?'); p += 2;
        } else { l1[n++] = '?'; ++p; while ((*p & 0xC0) == 0x80) ++p; }
    }
    l1[n] = 0;
    XStoreName(dpy_, win_, l1);
    XFlush(dpy_);
}

bool X11Window::create_image(Img& im, int w, int h, bool try_shm) {
    im = Img{};
    im.w = w; im.h = h;
    if (try_shm) {
        im.img = XShmCreateImage(dpy_, vis_, static_cast<unsigned>(depth_), ZPixmap, nullptr, &im.seg,
                                 static_cast<unsigned>(w), static_cast<unsigned>(h));
        if (im.img && im.img->bits_per_pixel == 32) {
            const usize bytes = static_cast<usize>(im.img->bytes_per_line) * static_cast<usize>(h);
            im.seg.shmid = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0600);
            if (im.seg.shmid >= 0) {
                im.seg.shmaddr = static_cast<char*>(shmat(im.seg.shmid, nullptr, 0));
                if (im.seg.shmaddr != reinterpret_cast<char*>(-1)) {
                    im.img->data = im.seg.shmaddr;
                    im.seg.readOnly = False;
                    g_x_error = false;
                    XErrorHandler old = XSetErrorHandler(x_error_trap);
                    const Status ok = XShmAttach(dpy_, &im.seg);
                    XSync(dpy_, False);
                    XSetErrorHandler(old);
                    shmctl(im.seg.shmid, IPC_RMID, nullptr);   // se libera al desadjuntar/morir
                    if (ok && !g_x_error) {
                        im.data = reinterpret_cast<u32*>(im.seg.shmaddr);
                        im.stride = im.img->bytes_per_line / 4;
                        im.shm = true;
                        return true;
                    }
                    shmdt(im.seg.shmaddr);
                } else {
                    shmctl(im.seg.shmid, IPC_RMID, nullptr);
                }
            }
        }
        if (im.img) { im.img->data = nullptr; XDestroyImage(im.img); }
        im = Img{};
        im.w = w; im.h = h;
        shm_ok_ = false;
        std::fprintf(stderr, "[x11] MIT-SHM no disponible: se usa XPutImage (más lento)\n");
    }
    // Ruta clásica: imagen en memoria propia alineada a 64 B.
    const int stride = (w + 15) & ~15;
    im.data = static_cast<u32*>(aligned_alloc_bytes(static_cast<usize>(stride) * static_cast<usize>(h) * 4));
    im.img = XCreateImage(dpy_, vis_, static_cast<unsigned>(depth_), ZPixmap, 0, reinterpret_cast<char*>(im.data),
                          static_cast<unsigned>(w), static_cast<unsigned>(h), 32, stride * 4);
    if (!im.img || im.img->bits_per_pixel != 32) {
        if (im.img) { im.img->data = nullptr; XDestroyImage(im.img); }
        aligned_free(im.data);
        im = Img{};
        return false;
    }
    im.stride = stride;
    im.shm = false;
    return true;
}

bool X11Window::create_images(int w, int h) {
    destroy_images();
    for (Img& im : img_)
        if (!create_image(im, max_(w, 1), max_(h, 1), shm_ok_)) return false;
    // Si la primera fue SHM y la segunda falló a SHM, ambas deben ser del mismo tipo.
    if (img_[0].shm != img_[1].shm) {
        destroy_images();
        shm_ok_ = false;
        for (Img& im : img_)
            if (!create_image(im, max_(w, 1), max_(h, 1), false)) return false;
    }
    cur_ = 0;
    return true;
}

void X11Window::destroy_images() {
    for (Img& im : img_) {
        if (!im.img) continue;
        if (im.shm) {
            wait_image(im, 0.2);
            XShmDetach(dpy_, &im.seg);
            XSync(dpy_, False);
            im.img->data = nullptr;
            XDestroyImage(im.img);
            shmdt(im.seg.shmaddr);
        } else {
            im.img->data = nullptr;
            XDestroyImage(im.img);
            aligned_free(im.data);
        }
        im = Img{};
    }
}

void X11Window::on_completion(const XEvent& ev) {
    const XShmCompletionEvent& ce = reinterpret_cast<const XShmCompletionEvent&>(ev);
    for (Img& im : img_)
        if (im.shm && im.seg.shmseg == ce.shmseg) im.busy = false;
}

void X11Window::wait_image(Img& im, double timeout_s) {
    if (!im.busy) return;
    const double deadline = now_sec() + timeout_s;
    XFlush(dpy_);
    while (im.busy) {
        XEvent ev;
        // Sólo saca del buffer los eventos ShmCompletion: la entrada queda para poll().
        if (XCheckTypedEvent(dpy_, shm_event_, &ev)) { on_completion(ev); continue; }
        if (now_sec() > deadline) { im.busy = false; break; }   // servidor sin respuesta: seguir
        pollfd pfd{ConnectionNumber(dpy_), POLLIN, 0};
        ::poll(&pfd, 1, 1);
    }
}

void X11Window::present(const render::Framebuffer& fb) {
    if (!dpy_ || !win_) return;
    const double t0 = now_sec();
    if (img_[0].w != win_w_ || img_[0].h != win_h_)
        if (!create_images(win_w_, win_h_)) return;
    Img& im = img_[cur_];
    wait_image(im, 0.1);
    const double t1 = now_sec();
    upscale_argb(fb.color.data(), fb.w, fb.h, fb.stride, im.data, im.w, im.h, im.stride, scale_, true, nt_);
    const double t2 = now_sec();
    if (im.shm) {
        XShmPutImage(dpy_, win_, gc_, im.img, 0, 0, 0, 0, static_cast<unsigned>(im.w), static_cast<unsigned>(im.h), True);
        im.busy = true;
    } else {
        XPutImage(dpy_, win_, gc_, im.img, 0, 0, 0, 0, static_cast<unsigned>(im.w), static_cast<unsigned>(im.h));
    }
    XFlush(dpy_);
    const double t3 = now_sec();
    cur_ ^= 1;
    stats_.wait_ms = (t1 - t0) * 1e3;
    stats_.upscale_ms = (t2 - t1) * 1e3;
    stats_.put_ms = (t3 - t2) * 1e3;
    stats_.total_ms = (t3 - t0) * 1e3;
    stats_.shm = im.shm;
    stats_.win_w = win_w_; stats_.win_h = win_h_;
    ++stats_.frames;
}

void X11Window::release_all(Input& in) {
    for (int b = 0; b < 3; ++b)
        if (in.mouse_down[b]) { in.mouse_down[b] = false; in.mouse_released[b] = true; }
    for (bool& k : in.key_down) k = false;
    in.shift = in.ctrl = in.alt = false;
    dead_ = 0;
}

void X11Window::handle(XEvent& ev, Input& in) {
    switch (ev.type) {
        case KeyPress: {
            char buf[32];
            KeySym ks = 0;
            const int n = XLookupString(&ev.xkey, buf, sizeof buf, &ks, nullptr);
            const KeySym base = XLookupKeysym(&ev.xkey, 0);
            const int k = map_keysym(base);
            if (k > KeyNone && k < KeyCount) { in.key_down[k] = true; in.key_pressed[k] = true; }
            if (ks >= XK_dead_grave && ks <= XK_dead_diaeresis) { dead_ = ks; break; }
            if (ev.xkey.state & ControlMask) { dead_ = 0; break; }   // atajos: sin texto
            for (int i = 0; i < n && in.text_len < 63; ++i) {
                u8 ch = static_cast<u8>(buf[i]);
                if (ch < 0x20 || ch == 0x7F) continue;
                if (dead_) {
                    const u8 comp = compose_dead(dead_, ch);
                    if (comp) ch = comp;
                    dead_ = 0;
                }
                in.text[in.text_len++] = static_cast<char>(ch);
            }
            in.text[in.text_len] = 0;
            break;
        }
        case KeyRelease: {
            // Sin autorepetición detectable: un KeyRelease seguido de un KeyPress idéntico es repetición.
            if (XEventsQueued(dpy_, QueuedAfterReading)) {
                XEvent nx;
                XPeekEvent(dpy_, &nx);
                if (nx.type == KeyPress && nx.xkey.keycode == ev.xkey.keycode && nx.xkey.time == ev.xkey.time) break;
            }
            const int k = map_keysym(XLookupKeysym(&ev.xkey, 0));
            if (k > KeyNone && k < KeyCount) in.key_down[k] = false;
            break;
        }
        case ButtonPress: {
            const unsigned b = ev.xbutton.button;
            if (b == Button4) { in.wheel += 1.0f; break; }
            if (b == Button5) { in.wheel -= 1.0f; break; }
            if (b < Button1 || b > Button3) break;
            const int mb = b == Button1 ? MouseLeft : (b == Button2 ? MouseMiddle : MouseRight);
            in.mouse_down[mb] = true;
            in.mouse_pressed[mb] = true;
            in.mouse_x = floor_div(ev.xbutton.x, scale_);
            in.mouse_y = floor_div(ev.xbutton.y, scale_);
            if (mb == MouseLeft) {
                const int dx = ev.xbutton.x - last_click_x_, dy = ev.xbutton.y - last_click_y_;
                if (last_click_ && ev.xbutton.time - last_click_ < 400 && dx * dx + dy * dy <= 16 * scale_ * scale_) {
                    in.double_click = true;
                    last_click_ = 0;               // un triple clic no genera dos dobles
                } else {
                    last_click_ = ev.xbutton.time;
                    last_click_x_ = ev.xbutton.x; last_click_y_ = ev.xbutton.y;
                }
            }
            break;
        }
        case ButtonRelease: {
            const unsigned b = ev.xbutton.button;
            if (b < Button1 || b > Button3) break;
            const int mb = b == Button1 ? MouseLeft : (b == Button2 ? MouseMiddle : MouseRight);
            if (in.mouse_down[mb]) in.mouse_released[mb] = true;
            in.mouse_down[mb] = false;
            break;
        }
        case MotionNotify: {
            const int x = ev.xmotion.x, y = ev.xmotion.y;
            if (pointer_valid_) { acc_dx_ += x - last_wx_; acc_dy_ += y - last_wy_; }
            last_wx_ = x; last_wy_ = y;
            pointer_valid_ = true;
            in.mouse_x = floor_div(x, scale_);
            in.mouse_y = floor_div(y, scale_);
            break;
        }
        case EnterNotify:
            last_wx_ = ev.xcrossing.x; last_wy_ = ev.xcrossing.y;
            pointer_valid_ = true;
            in.mouse_x = floor_div(last_wx_, scale_);
            in.mouse_y = floor_div(last_wy_, scale_);
            break;
        case LeaveNotify:
            // Fuera de la ventana y sin botones: el ratón "desaparece" (sin hover fantasma en la UI).
            if (ev.xcrossing.mode == NotifyNormal && !(in.mouse_down[0] || in.mouse_down[1] || in.mouse_down[2])) {
                in.mouse_x = in.mouse_y = -100000;
                pointer_valid_ = false;
            }
            break;
        case ConfigureNotify:
            if (ev.xconfigure.width != win_w_ || ev.xconfigure.height != win_h_) {
                win_w_ = max_(ev.xconfigure.width, 1);
                win_h_ = max_(ev.xconfigure.height, 1);
                const int nw = max_(win_w_ / scale_, 1), nh = max_(win_h_ / scale_, 1);
                if (nw != fb_w_ || nh != fb_h_) { fb_w_ = nw; fb_h_ = nh; size_changed_ = true; }
            }
            break;
        case ClientMessage:
            if (ev.xclient.message_type == wm_protocols_ && static_cast<Atom>(ev.xclient.data.l[0]) == wm_delete_) in.quit = true;
            break;
        case FocusOut:
            release_all(in);
            break;
        default:
            if (shm_event_ >= 0 && ev.type == shm_event_) on_completion(ev);
            break;
    }
}

void X11Window::poll(Input& in) {
    if (!dpy_) { in.quit = true; return; }
    while (XPending(dpy_)) {
        XEvent ev;
        XNextEvent(dpy_, &ev);
        handle(ev, in);
    }
    // Deltas en píxeles del framebuffer conservando el resto (sin perder movimiento al dividir).
    in.mouse_dx += acc_dx_ / scale_; acc_dx_ %= scale_;
    in.mouse_dy += acc_dy_ / scale_; acc_dy_ %= scale_;
    in.shift = in.key_down[KeyShift];
    in.ctrl = in.key_down[KeyCtrl];
    in.alt = in.key_down[KeyAlt];
    in.fb_w = fb_w_;
    in.fb_h = fb_h_;
    if (size_changed_) { in.resized = true; size_changed_ = false; }
}

void X11Window::set_fullscreen(bool on) {
    if (!dpy_ || !win_) return;
    XEvent e{};
    e.xclient.type = ClientMessage;
    e.xclient.window = win_;
    e.xclient.message_type = net_state_;
    e.xclient.format = 32;
    e.xclient.data.l[0] = on ? 1 : 0;             // _NET_WM_STATE_ADD / REMOVE
    e.xclient.data.l[1] = static_cast<long>(net_fs_);
    e.xclient.data.l[3] = 1;                      // fuente: aplicación normal
    XSendEvent(dpy_, RootWindow(dpy_, screen_), False, SubstructureRedirectMask | SubstructureNotifyMask, &e);
    XFlush(dpy_);
    fullscreen_ = on;
}

void X11Window::close() {
    if (!dpy_) return;
    destroy_images();
    if (gc_) { XFreeGC(dpy_, gc_); gc_ = nullptr; }
    if (win_) { XDestroyWindow(dpy_, win_); win_ = 0; }
    XCloseDisplay(dpy_);
    dpy_ = nullptr;
}

} // namespace

std::unique_ptr<Window> create_x11_window() {
    const char* disp = std::getenv("DISPLAY");
    if (!disp || !*disp) return nullptr;
    Display* d = XOpenDisplay(nullptr);
    if (!d) return nullptr;
    return std::make_unique<X11Window>(d);
}

int detect_pixel_scale() {
    if (const char* e = std::getenv("CFD_SCALE")) {
        const int s = std::atoi(e);
        if (s >= 1 && s <= 4) return s;
    }
    const char* disp = std::getenv("DISPLAY");
    if (!disp || !*disp) return 1;
    Display* d = XOpenDisplay(nullptr);
    if (!d) return 1;
    const int w = DisplayWidth(d, DefaultScreen(d));
    XCloseDisplay(d);
    return w >= 2560 ? 2 : 1;
}

} // namespace cfd::platform
