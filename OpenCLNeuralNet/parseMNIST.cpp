#include "parseMNIST.h"

#include <istream>
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void readMNIST(vector<float*> &inputs, vector<int*> &targets, int &inputSize)
{
    //Read the targets
    std::ifstream targetFile("data/train-labels.idx1-ubyte", std::ios::binary);
    if (targetFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        targetFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        targetFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);


        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            targetFile.read((char*)&temp,sizeof(temp));
            int* target = new int[10];
            std::fill_n(target,10,0);
            target[(int)temp] = 1;
            targets.push_back(target);

            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
    std::ifstream trainingFeatureFile ("data/train-images.idx3-ubyte", std::ios::binary);
    if (trainingFeatureFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        trainingFeatureFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        trainingFeatureFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        trainingFeatureFile.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        trainingFeatureFile.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        inputSize = n_rows * n_cols;
        for(int i=0;i<number_of_images;++i)
        {
            float* inputData = new float[n_rows*n_cols];
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    trainingFeatureFile.read((char*)&temp,sizeof(temp));
                    inputData[r*n_cols + c] = ((float)temp)/255.0;
                }
            }
            inputs.push_back(inputData);
            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
}

void readMNISTTest(vector<float*> &inputs, vector<int*> &targets, int &inputSize)
{
    //Read the targets
    std::ifstream targetFile("data/t10k-labels.idx1-ubyte", std::ios::binary);
    if (targetFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        targetFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        targetFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);


        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            targetFile.read((char*)&temp,sizeof(temp));
            int* target = new int[10];
            std::fill_n(target,10,0);
            target[(int)temp] = 1;
            targets.push_back(target);

            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
    std::ifstream trainingFeatureFile ("data/t10k-images.idx3-ubyte", std::ios::binary);
    if (trainingFeatureFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        trainingFeatureFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        trainingFeatureFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        trainingFeatureFile.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        trainingFeatureFile.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        inputSize = n_rows * n_cols;
        for(int i=0;i<number_of_images;++i)
        {
            float* inputData = new float[n_rows*n_cols];
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    trainingFeatureFile.read((char*)&temp,sizeof(temp));
                    inputData[r*n_cols + c] = ((float)temp)/255.0;
                }
            }
            inputs.push_back(inputData);
            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
}