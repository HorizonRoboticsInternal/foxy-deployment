import time
import pygame

from foxy.command_profile import ControllerCommandProfile


class GUI(object):
    def __init__(
        self,
        command_profile: ControllerCommandProfile,
        width: int = 1024,
        height: int = 768,
    ):
        pygame.init()
        self._resolution = (width, height)
        self._screen: pygame.Surface = pygame.display.set_mode((width, height))
        self._cmd = command_profile
        self._status_font = pygame.font.Font(None, 30)

    def space_to_continue(self):
        font = pygame.font.Font(None, 36)
        while True:
            pygame.event.get()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                break
            text = font.render("Press Space to Start ...", True, (0, 0, 0))
            self._screen.fill("white")
            self._screen.blit(text, (100, self._resolution[1] // 2))
            pygame.display.flip()
            time.sleep(0.1)

    def handle_once(self):
        """Handles the pygame events and rendering.

        Returns True to signal quit.
        """
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                return True

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            return True

        if keys[pygame.K_w]:
            self._cmd.cmd_x.alter(0.05)
        elif keys[pygame.K_s]:
            self._cmd.cmd_x.alter(-0.05)
        else:
            self._cmd.cmd_x.tween_to_default(0.05)

        if keys[pygame.K_a]:
            self._cmd.cmd_y.alter(0.01)
        elif keys[pygame.K_d]:
            self._cmd.cmd_y.alter(-0.01)
        else:
            self._cmd.cmd_y.tween_to_default(0.02)

        if keys[pygame.K_q]:
            self._cmd.cmd_yaw.alter(0.05)
        elif keys[pygame.K_e]:
            self._cmd.cmd_yaw.alter(-0.05)
        else:
            self._cmd.cmd_yaw.tween_to_default(0.05)

        if keys[pygame.K_UP]:
            self._cmd.cmd_ori_pitch.alter(0.01)
        elif keys[pygame.K_DOWN]:
            self._cmd.cmd_ori_pitch.alter(-0.01)
        else:
            self._cmd.cmd_ori_pitch.tween_to_default(0.02)

        if keys[pygame.K_RIGHT]:
            self._cmd.cmd_stance_width.alter(0.001)
        elif keys[pygame.K_LEFT]:
            self._cmd.cmd_stance_width.alter(-0.001)

        if keys[pygame.K_k]:
            self._cmd.cmd_height.alter(0.001)
        elif keys[pygame.K_l]:
            self._cmd.cmd_height.alter(-0.001)

        self._screen.fill("black")
        # Status
        text_lon = self._status_font.render(
            f"Lon [W, S]: {self._cmd.cmd_x.value:.2f}", True, (0, 255, 0)
        )
        self._screen.blit(text_lon, (50, 50))
        text_lat = self._status_font.render(
            f"Lat [A, D]: {self._cmd.cmd_y.value:.2f}", True, (0, 255, 0)
        )
        self._screen.blit(text_lat, (50, 80))
        pygame.display.flip()
        text_yaw = self._status_font.render(
            f"Yaw [Q, E]: {self._cmd.cmd_yaw.value:.2f}", True, (0, 255, 0)
        )
        self._screen.blit(text_yaw, (50, 110))
        text_pitch = self._status_font.render(
            f"Pitch [UP, DOWN]: {self._cmd.cmd_ori_pitch.value:.2f}", True, (0, 255, 0)
        )
        self._screen.blit(text_pitch, (50, 140))
        text_height = self._status_font.render(
            f"Height [K, L]: {self._cmd.cmd_height.value:.2f}", True, (0, 255, 0)
        )
        self._screen.blit(text_height, (50, 170))
        text_width = self._status_font.render(
            f"Width [K, L]: {self._cmd.cmd_stance_width.value:.2f}", True, (0, 255, 0)
        )
        self._screen.blit(text_width, (50, 200))
        pygame.display.flip()

        return False
