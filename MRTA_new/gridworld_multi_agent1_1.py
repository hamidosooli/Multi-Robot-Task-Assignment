import pygame as pg
import numpy as np
import pygame
import time


pygame.init()

# Constants
WIDTH = 800  # width of the environment (px)
HEIGHT = 800  # height of the environment (px)
TS = 10  # delay in msec
Col_num = 20  # number of columns
Row_num = 20  # number of rows

# define colors
bg_color = pg.Color(255, 255, 255)
line_color = pg.Color(128, 128, 128)
vfdr_color = pg.Color(8, 136, 8, 128)
vfds_color = pg.Color(255, 165, 0, 128)
vfdrs_color = pg.Color(173, 216, 230, 128)

def draw_grid(scr):
    '''a function to draw gridlines and other objects'''
    # Horizontal lines
    for j in range(Row_num + 1):
        pg.draw.line(scr, line_color, (0, j * HEIGHT // Row_num), (WIDTH, j * HEIGHT // Row_num), 2)
    # # Vertical lines
    for i in range(Col_num + 1):
        pg.draw.line(scr, line_color, (i * WIDTH // Col_num, 0), (i * WIDTH // Col_num, HEIGHT), 2)

    for x1 in range(0, WIDTH, WIDTH // Col_num):
        for y1 in range(0, HEIGHT, HEIGHT // Row_num):
            rect = pg.Rect(x1, y1, WIDTH // Col_num, HEIGHT // Row_num)
            pg.draw.rect(scr, bg_color, rect, 1)


def animate(rescue_team_traj, victims_traj, rescue_team_vfd, rescue_team_vfd_status, rescue_team_roles, env_map, wait_time):

    font = pg.font.SysFont('arial', 20)

    num_rescue_team = len(rescue_team_traj)
    num_victims = len(victims_traj)

    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))  # set up the screen
    pg.display.set_caption("gridworld")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()

    img_rescuer = pg.image.load('TurtleBot.png')
    img_mdf_r = pg.transform.scale(img_rescuer, (WIDTH // Col_num, HEIGHT // Row_num))

    img_rescuer_scout = pg.image.load('typhoon.jpg')
    img_mdf_rs = pg.transform.scale(img_rescuer_scout, (WIDTH // Col_num, HEIGHT // Row_num))

    img_scout = pg.image.load('Crazyflie.JPG')
    img_mdf_s = pg.transform.scale(img_scout, (WIDTH // Col_num, HEIGHT // Row_num))

    img_victim = pg.image.load('victim.png')
    img_mdf_victim = pg.transform.scale(img_victim, (WIDTH // Col_num, HEIGHT // Row_num))

    img_wall = pg.image.load('wall.png')
    img_mdf_wall = pg.transform.scale(img_wall, (WIDTH // Col_num, HEIGHT // Row_num))

    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    clock = pg.time.Clock()
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
        step = -1
        list_victims = np.arange(num_victims).tolist()
        list_rescue_team = np.arange(num_rescue_team).tolist()

        for rescue_team_stt, victims_stt in zip(np.moveaxis(rescue_team_traj, 0, -1),
                                                np.moveaxis(victims_traj, 0, -1)):

            for row in range(Row_num):
                for col in range(Col_num):
                    if env_map[row, col] == 1:
                        screen.blit(img_mdf_wall,
                                    (col * (WIDTH // Col_num),
                                     row * (HEIGHT // Row_num)))
            step += 1
            for num in list_rescue_team:
                if str(rescue_team_roles[num]) == "b'rs'":
                    vfd_color = vfdrs_color
                elif str(rescue_team_roles[num]) == "b'r'":
                    vfd_color = vfdr_color
                elif str(rescue_team_roles[num]) == "b's'":
                    vfd_color = vfds_color

                # rescuer visual field depth
                # vfd_j = 0
                # for j in range(int(max(rescue_team_stt[1, num] - rescue_team_vfd[num], 0)),
                #                int(min(Col_num, rescue_team_stt[1, num] + rescue_team_vfd[num] + 1))):
                #     vfd_i = 0
                #     for i in range(int(max(rescue_team_stt[0, num] - rescue_team_vfd[num], 0)),
                #                    int(min(Row_num, rescue_team_stt[0, num] + rescue_team_vfd[num] + 1))):
                #         if rescue_team_vfd_status[num][step][vfd_i, vfd_j]:
                #             rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                #                            (WIDTH // Col_num), (HEIGHT // Row_num))
                #             pg.draw.rect(screen, vfd_color, rect)
                #         vfd_i += 1
                #     vfd_j += 1

            # agents
            for num in list_rescue_team:
                if str(rescue_team_roles[num]) == "b'rs'":
                    img_mdf = img_mdf_rs
                elif str(rescue_team_roles[num]) == "b'r'":
                    img_mdf = img_mdf_r
                elif str(rescue_team_roles[num]) == "b's'":
                    img_mdf = img_mdf_s
                screen.blit(img_mdf,
                            (rescue_team_stt[1, num] * (WIDTH // Col_num),
                             rescue_team_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num), True, (0, 0, 0)),
                            (rescue_team_stt[1, num] * (WIDTH // Col_num),
                             rescue_team_stt[0, num] * (HEIGHT // Row_num)))

                # Stop showing finished agents
                # if (step >= 1 and
                #     (rescue_team_stt[:, num][0] == rescue_team_history[:, num][0] == rescue_team_traj[num, -1, 0] and
                #      rescue_team_stt[:, num][1] == rescue_team_history[:, num][1] == rescue_team_traj[num, -1, 1])):
                #     list_rescue_team.remove(num)

            for num in list_victims:
                screen.blit(img_mdf_victim, (victims_stt[1, num] * (WIDTH // Col_num),
                                             victims_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num), True, (0, 0, 0)),
                            (victims_stt[1, num] * (WIDTH // Col_num), victims_stt[0, num] * (HEIGHT // Row_num)))

                # Stop showing rescued victims
                # if step >= 1 and (victims_stt[:, num][0] == victims_history[:, num][0] == victims_traj[num, -1, 0] and
                #                   victims_stt[:, num][1] == victims_history[:, num][1] == victims_traj[num, -1, 1]):
                #     list_victims.remove(num)

            draw_grid(screen)
            pg.display.flip()
            pg.display.update()
            time.sleep(wait_time)  # wait between the shows

            for num in list_victims:
                screen.blit(bg, (victims_stt[1, num] * (WIDTH // Col_num),
                                 victims_stt[0, num] * (HEIGHT // Row_num)))

            for num in list_rescue_team:
                screen.blit(bg, (rescue_team_stt[1, num] * (WIDTH // Col_num),
                                 rescue_team_stt[0, num] * (HEIGHT // Row_num)))

                # rescuer visual field depths
                for j in range(int(max(rescue_team_stt[1, num] - rescue_team_vfd[num], 0)),
                               int(min(Row_num, rescue_team_stt[1, num] + rescue_team_vfd[num] + 1))):
                    for i in range(int(max(rescue_team_stt[0, num] - rescue_team_vfd[num], 0)),
                                   int(min(Col_num, rescue_team_stt[0, num] + rescue_team_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, bg_color, rect)

            victims_history = victims_stt
            rescue_team_history = rescue_team_stt

        for num in list_rescue_team:
            if str(rescue_team_roles[num]) == "b'rs'":
                img_mdf = img_mdf_rs
            elif str(rescue_team_roles[num]) == "b'r'":
                img_mdf = img_mdf_r
            elif str(rescue_team_roles[num]) == "b's'":
                img_mdf = img_mdf_s
            screen.blit(img_mdf, (rescue_team_traj[num, -1, 1] * (WIDTH // Col_num),
                                  rescue_team_traj[num, -1, 0] * (HEIGHT // Row_num)))
        for num in list_victims:
            screen.blit(img_mdf_victim, (victims_traj[num, -1, 1] * (WIDTH // Col_num),
                                         victims_traj[num, -1, 0] * (HEIGHT // Row_num)))

        draw_grid(screen)
        pg.display.flip()
        pg.display.update()
        run = False
    pg.quit()
