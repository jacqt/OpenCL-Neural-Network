#include "distortions.h"

float* getRandomDistortion(float* inputs) 
{
    //Returns several randomly translated image and scaled images
    int xTrans = (int) getRandomFloat(-2,2);
    int yTrans = (int) getRandomFloat(-2,2);
    float xScale = getRandomFloat(.94,1.06);
    float yScale = getRandomFloat(.94,1.06);
    float* distortedImagePre = scale(inputs, xScale, yScale);
    float* distortedImageFinal = translate(distortedImagePre, xTrans, yTrans);
    delete distortedImagePre;
    return distortedImageFinal;
}

float* translate(float* inputs, int x, int y)
{
    float* distortedImage = new float[28*28];
    int c = 0;
    int disp = x + y*28;
    for (int i = 0; i != 28; ++i)
    {
        for (int j = 0; j != 28; ++j)
        {
            if (c+disp < 0 || c + disp >= 28*28)
                distortedImage[c] = 0;
            else
                distortedImage[c] = inputs[c + disp];
            ++c;
        }
    }
    return distortedImage;
}

float* scale(float* inputs, float x, float y)
{
    float* distortedImage = new float[28*28];
    int c = 0;
    for (int i = 0; i != 28; ++i)
    {
        for (int j = 0; j != 28; ++j)
        {
            int row = float(i)*x;
            int column = float(j)*y;
            if (row > 27 || row < 0 || column > 27 || column < 0)
                distortedImage[c] = 0;
            else
                distortedImage[c] = inputs[28*row + column];
            ++c;
        }
    }
    return distortedImage;
}
