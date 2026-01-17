import pygame
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from maie.playground import Playground2D, HUD_FONT, HUD_FONT_COLOR
from maie.worlds.mars import MarsWorld, MarsConfig, MarsLayer
from maie.camera import RenderContext

# UI Constants
SIDEBAR_WIDTH = 200
BUTTON_HEIGHT = 30
BUTTON_MARGIN = 5
BG_COLOR = (30, 30, 30, 200) # Semi-transparent dark
ACTIVE_COLOR = (100, 200, 100)
INACTIVE_COLOR = (100, 100, 100)
HOVER_COLOR = (150, 150, 150)

class MarsPlayground(Playground2D):
    def __init__(self, world: MarsWorld):
        super().__init__(world, w=world.cfg.width, h=world.cfg.height, name="Mars Generator")
        self.world: MarsWorld = world
        self.show_ui = True
        self.buttons = []
        self._init_buttons()

    def _init_buttons(self):
        self.buttons = []
        y = 50
        for layer in MarsLayer:
            rect = pygame.Rect(10, y, SIDEBAR_WIDTH - 20, BUTTON_HEIGHT)
            self.buttons.append({"type": "layer", "layer": layer, "rect": rect, "label": layer.name})
            y += BUTTON_HEIGHT + BUTTON_MARGIN

        # Add Rain Toggle
        rect = pygame.Rect(10, y + 10, SIDEBAR_WIDTH - 20, BUTTON_HEIGHT)
        self.buttons.append({"type": "action", "action": "toggle_rain", "rect": rect, "label": "Rain: ON"})
        y += BUTTON_HEIGHT + BUTTON_MARGIN + 10

        # Add Speed Controls
        # Row for buttons
        btn_w = (SIDEBAR_WIDTH - 25) // 2
        r_dec = pygame.Rect(10, y + 30, btn_w, BUTTON_HEIGHT)
        r_inc = pygame.Rect(10 + btn_w + 5, y + 30, btn_w, BUTTON_HEIGHT)
        
        self.buttons.append({"type": "action", "action": "dec_sim", "rect": r_dec, "label": "-"})
        self.buttons.append({"type": "action", "action": "inc_sim", "rect": r_inc, "label": "+"})

    def _draw_ui(self, ctx: RenderContext):
        if not self.show_ui:
            return

        # Draw sidebar background
        s = pygame.Surface((SIDEBAR_WIDTH, ctx.screen.get_height()), pygame.SRCALPHA)
        s.fill(BG_COLOR)
        ctx.screen.blit(s, (0, 0))

        # Draw Buttons
        mx, my = pygame.mouse.get_pos()
        
        # Title
        title = HUD_FONT.render("MARS GENERATOR", True, (255, 200, 50))
        ctx.screen.blit(title, (10, 10))
        
        # Instruct
        info = HUD_FONT.render("TAB: Toggle UI", True, (150, 150, 150))
        ctx.screen.blit(info, (10, 30))

        for btn in self.buttons:
            rect = btn["rect"]
            
            is_active = False
            if btn["type"] == "layer":
                layer = btn["layer"]
                is_active = layer in self.world.enabled_layers
            elif btn["type"] == "action":
                if btn["action"] == "toggle_rain":
                    is_active = self.world.rain_enabled

            is_hover = rect.collidepoint(mx, my)

            color = ACTIVE_COLOR if is_active else INACTIVE_COLOR
            if is_hover:
                color = tuple(min(c + 50, 255) for c in color)

            # Text
            label = btn["label"]
            if btn["type"] == "action" and btn["action"] == "toggle_rain":
                label = f"Rain: {'ON' if self.world.rain_enabled else 'OFF'}"
                # Update active color dynamic
                is_active = self.world.rain_enabled
                color = ACTIVE_COLOR if is_active else (200, 50, 50)
                if is_hover: color = tuple(min(c + 50, 255) for c in color)
                pygame.draw.rect(ctx.screen, color, rect, border_radius=4)
            else:
                 pygame.draw.rect(ctx.screen, color, rect, border_radius=4)

            text_surf = HUD_FONT.render(label, True, (10, 10, 10) if is_active else (200, 200, 200))
            text_rect = text_surf.get_rect(center=rect.center)
            ctx.screen.blit(text_surf, text_rect)
            
            # Special Draw for Speed Label (hacky placement relative to buttons)
            if btn["type"] == "action" and btn["action"] == "dec_sim":
                # Draw label above
                lbl = f"Sim Speed: {self.world.cfg.sim_iterations}"
                ts = HUD_FONT.render(lbl, True, (200, 200, 200))
                # Centered in sidebar
                tr = ts.get_rect(center=(SIDEBAR_WIDTH // 2, rect.top - 15))
                ctx.screen.blit(ts, tr)

    def _handle_event(self, e: pygame.event.Event) -> None:
        # Custom UI handling
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_TAB:
                self.show_ui = not self.show_ui
                return # Consume tab

        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 1 and self.show_ui:
                mx, my = e.pos
                if mx < SIDEBAR_WIDTH:
                    # Click inside sidebar
                    for btn in self.buttons:
                        if btn["rect"].collidepoint(mx, my):
                            if btn["type"] == "layer":
                                layer = btn["layer"]
                                if layer in self.world.enabled_layers:
                                    self.world.enabled_layers.remove(layer)
                                else:
                                    self.world.enabled_layers.add(layer)
                                    # If enabling Water, restart the sim so user can see it
                                    if layer == MarsLayer.WATER:
                                        self.world.reset_hydrology()
                            elif btn["type"] == "action":
                                if btn["action"] == "toggle_rain":
                                    self.world.rain_enabled = not self.world.rain_enabled
                                elif btn["action"] == "inc_sim":
                                    self.world.cfg.sim_iterations += 10
                                elif btn["action"] == "dec_sim":
                                    self.world.cfg.sim_iterations = max(1, self.world.cfg.sim_iterations - 10)
                            return # Consumed
        
        # Fallback to default
        super()._handle_event(e)

    def run(self, fps: int = 30) -> None:
        # Override run to include UI drawing hook
        # Actually super().run() calls self._draw_hud/etc via DrawLayers
        # But Playground2D logic is a bit rigid in the snippet I saw.
        # It creates DrawLayers locally inside run() loop.
        # Safest is to just copy-paste the loop here or MonkeyPatch.
        # Let's copy-paste the main loop logic to be sure we control rendering order of UI.
        
        while True:
            for e in pygame.event.get():
                self._handle_event(e)
            
            if hasattr(self.world, 'update'):
                self.world.update()

            if self.on_frame is not None:
                self.on_frame(self)

            ms = pygame.mouse.get_pos()
            # camera logic is in super
            # but we need to create context
            
            # Recalculate context variables as in super
            from maie.camera import InputState
            inp = InputState(
                mouse_world=self.cam.screen_to_world(ms),
                mouse_screen=ms
            )
            ctx = RenderContext(
                screen=self.screen,
                camera=self.cam,
                input=inp,
                debug=False
            )

            # Draw World
            layers = self.world.get_layers(self.layers_to_draw)
            layers = sorted(layers, key=lambda x: x.z)

            from maie.playground import FILL_COLOR
            self.screen.fill(FILL_COLOR)
            
            for layer in layers:
                layer.draw(ctx)

            # Draw Extras
            if self.show_grid:
               self._draw_grid(ctx)
            if self.show_axes:
               self._draw_axes(ctx)

            # Draw Custom UI
            self._draw_hud(ctx) # Default HUD
            self._draw_ui(ctx)  # Our Sidebar

            pygame.display.flip()
            self.clock.tick(fps)

if __name__ == "__main__":
    cfg = MarsConfig()
    world = MarsWorld(cfg)
    pg = MarsPlayground(world)
    pg.run()
