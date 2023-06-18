import pygame
from systolic_injector import *

# Constants
SQUARE_SIZE = 50
SQUARE_PADDING = 30

def getTransformatedCoordinates(p : pygame.math.Vector2) -> pygame.math.Vector2:
    # Screen height and width
    global WIDTH
    global HEIGHT
    
    L = SQUARE_SIZE + SQUARE_PADDING

    p *= L
    # p.y = -p.y # y positive towards the top of the screen
    center = pygame.math.Vector2(WIDTH/2, HEIGHT/2)
    square_center_offset = pygame.math.Vector2(SQUARE_SIZE/2, SQUARE_SIZE/2)

    return p + center - square_center_offset

def drawPE(x, y, s):
    p = getTransformatedCoordinates( pygame.math.Vector2(x, y) )
    r = pygame.Rect(p.x, p.y, SQUARE_SIZE, SQUARE_SIZE)
    pygame.draw.rect(s, "white", r)

def draw_stuff(surface): 
    N1 = 3
    N2 = 3
    N3 = 3

    for i in range(1, N1+1):
        for j in range(1, N2+1):
            for k in range(1, N3+1):
                s = spaceTimeEquation([i,j,k], output_stationary)
                drawPE(s[0], s[1], surface)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600), flags = pygame.RESIZABLE)
    clock = pygame.time.Clock()
    running = True
    dt = 0
    
    global WIDTH
    global HEIGHT

    WIDTH = 800 * 2
    HEIGHT = 600 * 2
    main_surface = pygame.Surface( (WIDTH, HEIGHT) )
    main_surface.fill("yellow")

    x, y = ( (screen.get_width() - main_surface.get_width()) / 2, (screen.get_height() - main_surface.get_height()) / 2 )
    scale = 1
    speed = 1
    panning = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                if event.y == -1:
                    scale -= 0.1
                    if(scale >= 0.05): 
                        x += WIDTH * 0.05
                        y += HEIGHT * 0.05
                if event.y == +1:
                    scale += 0.1
                    x -= WIDTH * 0.05
                    y -= HEIGHT * 0.05
                scale = pygame.math.clamp(scale, 0.1, 100)

        screen.fill("purple")
        main_surface.fill("green")


        draw_stuff(main_surface)

        scaled_surface = pygame.transform.scale(main_surface, (WIDTH*scale, HEIGHT*scale) )
        screen.blit(scaled_surface, (x,y) )

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            y -= 1 * speed
        elif keys[pygame.K_s]:
            y += 1 * speed
        elif keys[pygame.K_a]:
            x -= 1 * speed
        elif keys[pygame.K_d]:
            x += 1 * speed

        mouse_btns = pygame.mouse.get_pressed()
        if mouse_btns[0] and not panning:
            panning = True
            m = pygame.mouse.get_pos()
        elif not mouse_btns[0] and panning:
            panning = False
        if panning:
            m_old = m
            m = pygame.mouse.get_pos()
            # print(m_old, " -> ", m , end = " : ")
            x_off = m[0] - m_old[0]
            y_off = m[1] - m_old[1]
            # print(x_off, ", ", y_off)
            x += x_off * speed
            y += y_off * speed
            
            
        
        pygame.display.flip() # update the screen
        clock.tick(60)

    pygame.quit()
