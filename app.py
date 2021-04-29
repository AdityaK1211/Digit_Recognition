import time
import cv2
import numpy as np
import pygame
import tensorflow as tf
from PIL import Image
from resizeimage import resizeimage

model = tf.keras.models.load_model('model/model.h5')
pygame.init()

display_width = 300
display_height = 300
radius = 10

black = (0, 0, 0)  # RGB
white = (255, 255, 255)

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Drawing Pad')


def digPredict(gameDisplay):
    # Processing the image into MNIST
    data = pygame.image.tostring(gameDisplay, 'RGBA')
    img = Image.frombytes('RGBA', (display_width, display_height), data)
    img = resizeimage.resize_cover(img, [28, 28])
    imgobj = np.asarray(img)
    imgobj = cv2.cvtColor(imgobj, cv2.COLOR_RGB2GRAY)
    (_, imgobj) = cv2.threshold(imgobj, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Predicting
    imgobj = imgobj / 255
    b = model.predict(np.reshape(imgobj, [1, imgobj.shape[0], imgobj.shape[1], 1]))
    ans = np.argmax(b)
    print("Predicted Value: ", ans)
    return ans


def textObjects(text, font):
    textSurface = font.render(text, True, white)
    return textSurface, textSurface.get_rect()


def message_display(text, locx, locy, size):
    largeText = pygame.font.Font('freesansbold.ttf', size)  # Font(font name, font size)
    TextSurf, TextRec = textObjects(text, largeText)
    TextRec.center = (locx, locy)
    gameDisplay.blit(TextSurf, TextRec)
    pygame.display.update()


def gameLoop():
    gameExit = False
    gameDisplay.fill(black)
    pygame.display.flip()
    tick = 0
    tock = 0
    startDraw = False
    while not gameExit:
        if tock - tick >= 2 and startDraw:
            predVal = digPredict(gameDisplay)
            gameDisplay.fill(black)
            message_display("Predicted Value: " + str(predVal), int(display_width / 2), int(display_height / 2), 20)
            time.sleep(2)  # sleep for 2 seconds
            gameDisplay.fill(black)
            pygame.display.flip()
            tick = 0
            tock = 0
            startDraw = False
            continue

        tock = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
        if pygame.mouse.get_pressed()[0]:
            spot = pygame.mouse.get_pos()
            pygame.draw.circle(gameDisplay, white, spot, radius)
            pygame.display.flip()
            tick = time.perf_counter()
            startDraw = True


gameLoop()
pygame.quit()
