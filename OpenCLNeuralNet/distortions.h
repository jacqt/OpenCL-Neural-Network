#ifndef DISTORTIONS_H
#define DISTORTIONS_H

#include "include.h"
#include "layer.h"

float* getRandomDistortion(float* inputs);
float* translate(float* inputs, int x, int y);
float* scale(float* inputs, float x, float y);

#endif