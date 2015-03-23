#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand


using namespace std;


int main()
{
    std::srand ( unsigned ( std::time(0) ) );

    //std::default_random_engine generator(std::time(0));
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5,0.5);

    const int num_hidden_unit = 2;
    const int num_train_data = 4;
    const int num_test_data = 10;


    // entire train data
    double train_data_input[num_train_data][2];
    double train_flag[num_train_data];

    double ingta = 0.5;
    double input_hidden_v[3][num_hidden_unit];
    double hidden_output_v[num_hidden_unit+1];

    double alpha = 0.4;
    double delta_input_hidden_v[3][num_hidden_unit] = {0};
    double delta_hidden_output_v[num_hidden_unit+1] = {0};

    hidden_output_v[0] = distribution(generator);
    for (int j = 0; j < num_hidden_unit; ++j) {
        input_hidden_v[0][j] = distribution(generator);
        input_hidden_v[1][j] = distribution(generator);
        input_hidden_v[2][j] = distribution(generator);

        hidden_output_v[j+1] = distribution(generator);
    }


    double input_y[2];
    double input_v[num_hidden_unit];

    double hidden_y[num_hidden_unit];
    double hidden_v;

    double output_y;
    double output_d;

    double delta_output;
    double delta_hidden[num_hidden_unit];

    std::ifstream infile("/home/york/nn/xor_train.txt");
    for (int k=0; k < num_train_data; ++k) {
        infile >> train_data_input[k][0] >> train_data_input[k][1];
        infile >> train_flag[k];
    }

    std::vector<int> myvector;

    // set some values:
    for (int i=0; i<num_train_data; ++i) myvector.push_back(i);

    int label;
    for(int k=0; k<1000; ++k) {
        // using built-in random generator:
        std::random_shuffle ( myvector.begin(), myvector.end() );
        std::vector<int>::iterator it=myvector.begin();
        double total_dist_y = 0;
        for (int i = 0; i < num_train_data; ++i) {
            // one input
            input_y[0] = train_data_input[*it][0];
            input_y[1] = train_data_input[*it][1];
            label = train_flag[*it];
            ++it;

            output_d = label;

            /*************************
            * forwarding process
            **************************/
            // calculate local of input
            for (int j = 0; j < num_hidden_unit; ++j) {
                input_v[j] = (input_hidden_v[0][j] + input_hidden_v[1][j] * input_y[0] + input_hidden_v[2][j] * input_y[1]);
                // input of hidden
                hidden_y[j] = 1.0 / (1 + exp(-input_v[j]));
            }

            // calculate local of hidden
            hidden_v = hidden_output_v[0];
            for (int j = 0; j < num_hidden_unit; ++j) {
                hidden_v += (hidden_output_v[j + 1] * hidden_y[j]);
            }

            // input of last layer, i.e. output
            output_y = 1.0 / (1 + exp(-hidden_v));


            /**************************
            * bp process
            ***************************/

            delta_output = (output_d - output_y) * (output_y) * (1 - output_y);
            total_dist_y += abs(delta_output);

            for (int j = 0; j < num_hidden_unit; ++j) {
                delta_hidden[j] = (hidden_y[j]) * (1 - hidden_y[j]) * (delta_output * hidden_output_v[j + 1]);
            }

            for (int j = 0; j <= num_hidden_unit; ++j) {
                delta_hidden_output_v[j] = (alpha * delta_hidden_output_v[j] + ingta * (delta_output * hidden_y[j]));
                hidden_output_v[j] += delta_hidden_output_v[j];
            }


            for (int j = 0; j < num_hidden_unit; ++j) {
                delta_input_hidden_v[0][j] = (alpha * delta_input_hidden_v[0][j] + ingta * delta_hidden[j]);
                input_hidden_v[0][j] += delta_input_hidden_v[0][j];

                delta_input_hidden_v[1][j] = (alpha * delta_input_hidden_v[1][j] + ingta * (delta_hidden[j] * input_y[0]));
                input_hidden_v[1][j] += delta_input_hidden_v[1][j];

                delta_input_hidden_v[2][j] = (alpha * delta_input_hidden_v[2][j] + ingta * (delta_hidden[j] * input_y[1]));
                input_hidden_v[2][j] += delta_input_hidden_v[2][j];
            }
        }

        //cout << total_dist_y << endl;
        if (abs(total_dist_y) < 1e-4) {
            cout << "done" << endl;
            break;
        }
    }

    // cout trained data
    cout << "the trained parameter:\n";
    cout << input_hidden_v[0][0] << " " << input_hidden_v[0][1] << endl;
    cout << input_hidden_v[1][0] << " " << input_hidden_v[1][1] << endl;
    cout << input_hidden_v[2][0] << " " << input_hidden_v[2][1] << endl;
    cout << hidden_output_v[0] << " " << hidden_output_v[1] << " " << hidden_output_v[2] << endl;
    cout << endl;

    std::ifstream testfile("/home/york/nn/xor_test.txt");

    int total_error = 0;
    for (int i = 0; i < num_test_data; ++i) {
        testfile >> input_y[0] >> input_y[1];
        testfile >> label;

        // calculate local of input
        for (int j = 0; j < num_hidden_unit; ++j) {
            input_v[j] = (input_hidden_v[0][j] + input_hidden_v[1][j] * input_y[0] + input_hidden_v[2][j] * input_y[1]);
            // input of hidden
            hidden_y[j] = 1.0 / (1 + exp(-input_v[j]));
        }

        // calculate local of hidden
        hidden_v = hidden_output_v[0];
        for (int j = 0; j < num_hidden_unit; ++j) {
            hidden_v += (hidden_output_v[j + 1] * hidden_y[j]);
        }

        // input of last layer, i.e. output
        output_y = 1.0 / (1 + exp(-hidden_v));

        total_error += ((output_y < 0.5 ? 0 : 1) != label);
        cout << output_y << " " << label << endl;
    }


    cout << total_error;

    return 0;
};