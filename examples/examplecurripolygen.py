from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from shapeset.curridata import *
from shapeset.buildfeaturespolygon import *
from shapeset.polygongen import *

n = 1
m = 1

genparams = {'inv_chance': 0.5, 'img_shape': (128, 128), 'n_vert_list': [3, 4, 20], 'fg_min': 0.55, 'fg_max': 1.0,
             'bg_min': 0.0, 'bg_max': 0.45, 'rot_min': 0.0, 'rot_max': 1.0, 'pos_min': 0, 'pos_max': 1,
             'scale_min': 0.2, 'scale_max': 0.8, 'rotation_resolution': 255,
             'nb_poly_max': 2, 'nb_poly_min': 1, 'overlap_max': 0.5, 'poly_type': 2, 'rejectionmax': 50,
             'overlap_bool': True}

# genparams2 = {'poly_type' :2,'rot_max' : 1}

datagenerator = Polygongen
funclist = [buildimage, buildedgesangle, builddepthmap, buildidentity, buildsegmentation, output, buildedgesanglec, output_angles,
            output_as_Shapeset3x2_categorical, output_as_ShapesetNxM_categorical]
dependencies = [None, {'segmentation': 4}, None, None, {'depthmap': 2}, None, {'segmentation': 4}, None, None, None]
funcparams = {'neighbor': 'V8', 'gaussfiltbool': False, 'sigma': 0.5, 'size': 5, 'neg': True}
batchsize = n * m
seed = 0

curridata = Curridata(datagenerator, genparams, funclist, dependencies, funcparams, batchsize, seed,
                      generatorReturnsBatch=True, feature_input=0, feature_output=7)
# here you need to hard code the targets and inputs property field
Curridata.image = property(functools.partial(Curridata.getter, i=0))
Curridata.edges = property(functools.partial(Curridata.getter, i=1))
Curridata.depth = property(functools.partial(Curridata.getter, i=2))
Curridata.identity = property(functools.partial(Curridata.getter, i=3))
Curridata.segmentation = property(functools.partial(Curridata.getter, i=4))
Curridata.output = property(functools.partial(Curridata.getter, i=5))
Curridata.edgesc = property(functools.partial(Curridata.getter, i=6))
Curridata.output_angles = property(functools.partial(Curridata.getter, i=7))
Curridata.output_as_Shapeset3x2_categorical = property(functools.partial(Curridata.getter, i=8))
Curridata.output_as_ShapesetNxM_categorical= property(functools.partial(Curridata.getter, i=9))
# curridata.changegenparam(genparams2)

# ------------------------------------------------------------------------------------------------

# To draw with pygame

pygame.display.init()

screen = pygame.display.set_mode((n * genparams['img_shape'][0] * 2, m * genparams['img_shape'][1] * 6), 0, 8)

anglcolorpalette = [(0, 0, 0)] + [(0, 0, 255)] + [(0, 255, 0)] + [(255, 0, 0)] + [(255, 255, 0)] + \
                   [(x, x, x) for x in range(5, 256)]
screen.set_palette(anglcolorpalette)

iteration = 0
nmult = 4
if funcparams['neighbor'] is 'V4':
    nmult = 2


def showresult(it):
    batch_data = curridata.next()

    xvalid = (np.reshape((curridata.image + 1) * 0.5 * 255,
                            (batchsize, genparams['img_shape'][0], genparams['img_shape'][1])) / 255.0 * 250 + 5)
    yvalid = np.reshape((curridata.edges + 1) * 0.5, (batchsize, 4, genparams['img_shape'][0], genparams['img_shape'][1]))
    zvalid = (np.reshape((curridata.depth + 1) * 0.5 * 255,
                            (batchsize, genparams['img_shape'][0], genparams['img_shape'][1])) / 255.0 * 250 + 5)
    wvalid = np.reshape((curridata.identity + 1) * 0.5,
                           (batchsize, len(genparams['n_vert_list']), genparams['img_shape'][0], genparams['img_shape'][1]))
    svalid = np.reshape((curridata.segmentation + 1) * 0.5 * 255,
                           (batchsize, genparams['img_shape'][0] * nmult, genparams['img_shape'][1]))
    tvalid = (np.reshape((curridata.edgesc + 1) * 0.5 * 255,
                            (batchsize, 4, genparams['img_shape'][0], genparams['img_shape'][1])) / 255.0 * 250 + 5)

    for j in range(batchsize):
        xi = (j / m) * genparams['img_shape'][0] * 2
        yi = (j - (j / m) * m) * genparams['img_shape'][1] * 6
        print(xi, yi)

        new = pygame.surfarray.make_surface(xvalid[j, :, :])
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi, yi))

        ytmp = (yvalid[j, 2, :, :] * yvalid[j, 3, :, :]) * 4 + (yvalid[j, 0, :, :] * yvalid[j, 3, :, :]) * 1 + (
                yvalid[j, 0, :, :] * yvalid[j, 1, :, :]) * 2 + (yvalid[j, 1, :, :] * yvalid[j, 2, :, :]) * 3
        new = pygame.surfarray.make_surface(ytmp)
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + genparams['img_shape'][0], yi + 0))

        new = pygame.surfarray.make_surface(zvalid[j, :, :])
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + 0, yi + genparams['img_shape'][1]))

        for idx in range(wvalid.shape[1]):
            ytmp = + wvalid[j, idx, :, :] * (idx + 1)
        new = pygame.surfarray.make_surface(ytmp)
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + genparams['img_shape'][0], yi + genparams['img_shape'][1]))

        new = pygame.surfarray.make_surface(svalid[j, :genparams['img_shape'][0] * 2 - 1, :])
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + 0, yi + genparams['img_shape'][1] * 2))

        if nmult != 2:
            new = pygame.surfarray.make_surface(svalid[j, genparams['img_shape'][0] * 2:genparams['img_shape'][0] * 4 - 1, :])
            new.set_palette(anglcolorpalette)
            screen.blit(new, (xi + 0, yi + genparams['img_shape'][1] * 3))

        new = pygame.surfarray.make_surface(tvalid[j, 2, :, :])
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + 0, yi + genparams['img_shape'][1] * 4))

        new = pygame.surfarray.make_surface(tvalid[j, 0, :, :])
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + genparams['img_shape'][0], yi + genparams['img_shape'][1] * 4))

        new = pygame.surfarray.make_surface(tvalid[j, 1, :, :])
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + 0, yi + genparams['img_shape'][1] * 5))

        new = pygame.surfarray.make_surface(tvalid[j, 3, :, :])
        new.set_palette(anglcolorpalette)
        screen.blit(new, (xi + genparams['img_shape'][0], yi + genparams['img_shape'][1] * 5))

        pygame.display.update()
        print('\noutput')
        print (curridata.output[j, :])
        # print('\noutput_as_angles')
        # print (curridata.output_angles)
        # print('\noutput_as_Shapeset3x2_categorical')
        # print (curridata.output_as_Shapeset3x2_categorical)
        # print('\noutput_as_ShapesetMxN_categorical')
        # print (curridata.output_as_ShapesetNxM_categorical)
        # raw_input("Please press Enter")
    it += 1
    print (it)
    return it
    # raw_input("Please press Enter")


# ------------------------------------------------------------------------------------------------

# to test with timeit


def genex(curridata, n, mode=1):
    it = 0
    while it < n:
        if mode:
            curridata.image
            curridata.segmentation
            curridata.edges
            curridata.depth
            curridata.identity
            curridata.output
        curridata.next()
        it += 1
        # print it


# tim1=Timer("genex(curridata,1000,1)","from __main__ import genex, curridata ; gc.enable()").timeit(1)
# tim2=Timer("genex(curridata,1000,0)","from __main__ import genex, curridata ; gc.enable()").timeit(1)

# ------------------------------------------------------------------------------------------------

# Fix Python 2.x.
try:
    # in python 2.x we want raw_input
    get_input = raw_input
except NameError:
    # in python 3.x we want input
    get_input = input

# curridata.gen.rot_min=0
for i in range(100):
    iteration = showresult(iteration)
    get_input('Press enter to continue generation')

# pygame.display.quit()
