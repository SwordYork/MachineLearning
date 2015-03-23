/******************************
* Author: york
* Email: z@slinuxer.com
* Date: 2015/03/22
* Description:
*    Homework of Neural Network. This program is a NN to classify spiral data
******************************/
#include <iostream>
#include <fstream>
#include <cmath>
#include <array>
#include "nn_batch.h"
#include "nn_seq.h"


using std::cout;
using std::endl;


int main()
{

    cout << "testing batch nn....." << endl;
    nn_batch_spiral();
    cout << endl;
    cout << endl;
    cout << endl;

    cout << "testing seq nn....." << endl;
    nn_seq_spiral();
    cout << endl;



    return 0;
};