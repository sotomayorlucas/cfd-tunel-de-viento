// ============================================================================
//  platform/platform.hpp — ventana, entrada y presentación (contrato público).
//
//  Backends: X11 + MIT-SHM (XShmPutImage, memoria compartida con el servidor
//  X → sin copias por el socket) y "headless" (sin ventana: para tests,
//  benchmarks y capturas PNG).
//  pixel_scale: la ventana mide fb*scale píxeles; el escalado entero (2×2 por
//  píxel en pantallas HiDPI 3840×2400) se hace con AVX2 al presentar.
//  Todas las coordenadas de ratón que entrega Input están en píxeles del
//  FRAMEBUFFER (ya divididas por pixel_scale).
// ============================================================================
#pragma once

#include "../core/mathx.hpp"          // framebuffer.hpp usa saturate/Vec3 sin incluirlo
#include "../render/framebuffer.hpp"
#include <memory>

namespace cfd::platform {

enum Key : int {
    KeyNone = 0,
    // 32..126 = ASCII imprimible (minúsculas para letras)
    KeyEscape = 256, KeyEnter, KeyTab, KeyBackspace, KeyDelete, KeyInsert,
    KeyLeft, KeyRight, KeyUp, KeyDown, KeyHome, KeyEnd, KeyPageUp, KeyPageDown,
    KeyF1, KeyF2, KeyF3, KeyF4, KeyF5, KeyF6, KeyF7, KeyF8, KeyF9, KeyF10, KeyF11, KeyF12,
    KeyShift, KeyCtrl, KeyAlt,
    KeyCount = 512
};

enum MouseButton : int { MouseLeft = 0, MouseMiddle = 1, MouseRight = 2 };

struct Input {
    int mouse_x = 0, mouse_y = 0;       // píxeles del framebuffer
    int mouse_dx = 0, mouse_dy = 0;     // delta desde el cuadro anterior
    bool mouse_down[3] = {};
    bool mouse_pressed[3] = {};         // flanco de bajada este cuadro
    bool mouse_released[3] = {};
    bool double_click = false;          // doble clic izquierdo este cuadro
    float wheel = 0;                    // +1 por muesca hacia arriba
    bool key_down[KeyCount] = {};
    bool key_pressed[KeyCount] = {};    // incluye autorepetición
    bool shift = false, ctrl = false, alt = false;
    char text[64] = {};                 // caracteres tecleados este cuadro (Latin-1), terminado en 0
    int text_len = 0;
    bool quit = false;                  // cerrar ventana / Escape doble, etc.
    bool resized = false;
    int fb_w = 0, fb_h = 0;             // tamaño actual del framebuffer

    // Borra los flancos (pressed/released/wheel/text/deltas) al empezar un cuadro.
    void begin_frame() {
        mouse_dx = mouse_dy = 0;
        for (bool& b : mouse_pressed) b = false;
        for (bool& b : mouse_released) b = false;
        for (bool& b : key_pressed) b = false;
        double_click = false;
        wheel = 0; text_len = 0; text[0] = 0; resized = false;
    }
};

// Estadísticas de presentación (extensión compatible del contrato, módulo ui+plataforma).
struct PresentStats {
    double upscale_ms = 0;      // escalado entero AVX2 (paralelo) del último present
    double put_ms = 0;          // XShmPutImage/XPutImage + flush
    double wait_ms = 0;         // espera a ShmCompletion (búfer aún leído por el servidor)
    double total_ms = 0;        // present() completo
    u64 frames = 0;
    bool shm = false;           // true = MIT-SHM, false = XPutImage por el socket (o headless)
    int win_w = 0, win_h = 0;   // tamaño de la ventana en píxeles físicos
};

class Window {
public:
    virtual ~Window() = default;
    // Crea la ventana. fb_w×fb_h = tamaño del framebuffer; la ventana mide ×pixel_scale.
    virtual bool open(const char* title, int fb_w, int fb_h, int pixel_scale) = 0;
    // Procesa eventos pendientes (no bloquea). Actualiza Input (llamar a input.begin_frame() antes).
    virtual void poll(Input& input) = 0;
    // Muestra el framebuffer (escala entera si pixel_scale > 1). Si el tamaño de ventana cambió,
    // Input::resized/fb_w/fb_h lo indican y la app redimensiona su Framebuffer.
    virtual void present(const render::Framebuffer& fb) = 0;
    virtual void set_title(const char* title) = 0;
    virtual int pixel_scale() const = 0;
    virtual bool is_headless() const = 0;
    // --- Extensiones compatibles (implementación por defecto) ---
    virtual PresentStats present_stats() const { return {}; }
    virtual void set_fullscreen(bool on) { (void)on; }       // _NET_WM_STATE_FULLSCREEN en X11
    virtual bool fullscreen() const { return false; }
};

// nullptr si no hay servidor X (DISPLAY) disponible.
std::unique_ptr<Window> create_x11_window();
// Ventana ficticia: present() no hace nada salvo contar cuadros (la app guarda PNGs aparte).
std::unique_ptr<Window> create_headless_window();

// Pixel scale recomendado según la resolución de la pantalla (2 si ≥ 2560 px de ancho). 1 si no hay X.
// La variable de entorno CFD_SCALE=n (1..4) lo fuerza.
int detect_pixel_scale();

// Escalado entero (vecino más próximo) src (sw×sh) → dst, `scale` ∈ [1,8]. Escribe exactamente
// min(dw, sw·scale) × min(dh, sh·scale) píxeles; el resto de dst (si la ventana es mayor) se pone
// a negro. AVX2: 8 píxeles → 16 con unpack + permute2x128, filas repetidas desde registros y
// stores no temporales si dst está alineado a 32 B (la imagen va al servidor X, no a nuestra caché).
// parallel=true reparte filas en el pool. Expuesto para tests/benchmarks.
void upscale_argb(const u32* src, int sw, int sh, int sstride, u32* dst, int dw, int dh, int dstride,
                  int scale, bool parallel = true, bool nontemporal = true);

} // namespace cfd::platform
