//
// Created by york on 3/22/15.
//

#include <random>
#include <ctime>        // std::time
#include <iostream>
#include <fstream>
#include <algorithm>    // std::random_shuffle
#include "nn_batch.h"
#include "common.h"

using std::cout;
using std::endl;

void nn_batch_spiral()
{
    unsigned int seed = static_cast<unsigned int>(std::time(0));
    std::srand(seed);
    //std::default_random_engine generator;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);

    // the stop critical
    /*************
    * |last_error - current_error| / last_error < eps
    **************/
    const double eps = 1e-5;
    const int max_iter = 8000;

    // learning rate
    // adapt during epoch
    double ingta = 4;
    double alpha = 0.5;

    // entire train data set
    // random shuffle during each epoch
    double train_data_input[num_train_data][2];
    double train_flag[num_train_data];

    // input to hidden weight
    // hidden to output weight
    double input_hidden_v[3][num_hidden_unit];
    double hidden_output_v[num_hidden_unit+1];
    double input_hidden_u[2][num_hidden_unit];
    double hidden_output_u[num_hidden_unit];

    // record delta of last step
    // it will be more stable to update
    double delta_input_hidden_v[3][num_hidden_unit] = {0};
    double delta_hidden_output_v[num_hidden_unit+1] = {0};
    double delta_input_hidden_u[2][num_hidden_unit] = {0};
    double delta_hidden_output_u[num_hidden_unit] = {0};

    // generate initial weight
    // it's better to uniform between (-0.5, 0.5)
    hidden_output_v[0] = distribution(generator);
    for (int j = 0; j < num_hidden_unit; ++j) {
        input_hidden_v[0][j] = distribution(generator);
        input_hidden_v[1][j] = distribution(generator);
        input_hidden_v[2][j] = distribution(generator);

        hidden_output_v[j+1] = distribution(generator);

        input_hidden_u[0][j] = distribution(generator);
        input_hidden_u[1][j] = distribution(generator);
        hidden_output_u[j] = distribution(generator);
    }


    // some middle result
    double input_y[2];
    double input_v[num_hidden_unit];

    double hidden_y[num_hidden_unit];
    double hidden_v;

    double output_y;
    double output_d;

    // delta used in BP process
    double delta_output;
    double delta_hidden[num_hidden_unit];

    // read entire data set
    std::ifstream infile("two_spiral_train.txt");
    for (int k=0; k < num_train_data; ++k) {
        infile >> train_data_input[k][0] >> train_data_input[k][1];
        infile >> train_flag[k];
    }



    std::clock_t start_clock;
    start_clock = std::clock();

    // to generate random shuffle
    std::vector<int> shuffle_vector;
    for (int i=0; i<num_train_data; ++i) shuffle_vector.push_back(i);

    double last_error = 0;
    double current_error = 0;

    for(int k = 0; k < max_iter; ++k) {
        // using built-in random generator:
        std::random_shuffle(shuffle_vector.begin(), shuffle_vector.end());
        std::vector<int>::iterator it = shuffle_vector.begin();

        // to test stop critical
        last_error = current_error;
        current_error = 0;

        // adjust learning rate
        if (k % 500 == 0 && k != 0) {
            ingta /= 2;
            alpha /= 8;
        }

        // one epoch
        for (int i = 0; i < num_train_data; ++i) {
            // one input
            input_y[0] = train_data_input[*it][0];
            input_y[1] = train_data_input[*it][1];
            output_d = train_flag[*it];
            ++it;

            /*************************
            * forwarding process
            **************************/
            // calculate local of input
            for (int j = 0; j < num_hidden_unit; ++j) {
                input_v[j] = (input_hidden_v[0][j] + input_hidden_v[1][j] * input_y[0] + input_hidden_v[2][j] * input_y[1]);
                input_v[j] += (input_hidden_u[0][j] * input_y[0] * input_y[0] + input_hidden_u[1][j] * input_y[1] * input_y[1]);
                // input of hidden
                hidden_y[j] = 1.0 / (1 + exp(-input_v[j]));
            }

            // calculate local of hidden
            hidden_v = hidden_output_v[0];
            for (int j = 0; j < num_hidden_unit; ++j) {
                hidden_v += (hidden_output_v[j + 1] * hidden_y[j]);
                hidden_v += (hidden_output_u[j] * hidden_y[j] * hidden_y[j]);
            }

            // input of last layer, i.e. output
            output_y = 1.0 / (1 + exp(-hidden_v));
            current_error += 0.5 * (output_d - output_y) * (output_d - output_y);

            /**************************
            * BP process
            ***************************/
            delta_output = (output_d - output_y) * (output_y) * (1 - output_y);

            // delta of hidden using BP
            for (int j = 0; j < num_hidden_unit; ++j) {
                delta_hidden[j] = (hidden_y[j]) * (1 - hidden_y[j]) * (delta_output *
                        (hidden_output_v[j + 1] + 2 * hidden_output_u[j] * hidden_y[j]));
            }


            // update hidden to output weight v

            for (int j = 0; j <= num_hidden_unit; ++j) {
                delta_hidden_output_v[j] += ingta * (delta_output * hidden_y[j]);
            }

            // update hidden to output weight u
            for (int j = 0; j < num_hidden_unit; ++j) {
                delta_hidden_output_u[j] += ingta * delta_output * hidden_y[j] * hidden_y[j];
            }

            // update input to hidden weight
            for (int j = 0; j < num_hidden_unit; ++j) {
                delta_input_hidden_v[0][j] += ingta * delta_hidden[j];

                delta_input_hidden_v[1][j] += ingta * (delta_hidden[j] * input_y[0]);

                delta_input_hidden_v[2][j] += ingta * (delta_hidden[j] * input_y[1]);

                delta_input_hidden_u[0][j] += ingta * delta_hidden[j] * input_y[0] * input_y[0];

                delta_input_hidden_u[1][j] += ingta * delta_hidden[j] * input_y[1] * input_y[1];
            }
        }

        for (int j = 0; j <= num_hidden_unit; ++j) {
            hidden_output_v[j] += delta_hidden_output_v[j] / num_train_data;
            delta_hidden_output_v[j] *= alpha / num_train_data;
        }

        // update hidden to output weight u
        for (int j = 0; j < num_hidden_unit; ++j) {
            hidden_output_u[j] += delta_hidden_output_u[j] / num_train_data;
            delta_hidden_output_u[j] *= alpha / num_train_data;
        }

        // update input to hidden weight
        for (int j = 0; j < num_hidden_unit; ++j) {
            input_hidden_v[0][j] += delta_input_hidden_v[0][j] / num_train_data;
            delta_input_hidden_v[0][j] *= alpha / num_train_data;

            input_hidden_v[1][j] += delta_input_hidden_v[1][j] / num_train_data;
            delta_input_hidden_v[1][j] *= alpha / num_train_data;

            input_hidden_v[2][j] += delta_input_hidden_v[2][j] / num_train_data;
            delta_input_hidden_v[2][j] *= alpha / num_train_data;

            input_hidden_u[0][j] += delta_input_hidden_u[0][j] / num_train_data;
            delta_input_hidden_u[0][j] *= alpha / num_train_data;


            input_hidden_u[1][j] += delta_input_hidden_u[1][j] / num_train_data;
            delta_input_hidden_u[1][j] *= alpha / num_train_data;
        }

        if (std::abs(last_error-current_error) * 1.0 / last_error < eps) {
            cout << k << " epoch done" << endl;
            break;
        }
    }

    cout << "training time:" << (std::clock() - start_clock) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

    // cout trained data
    std::ofstream outfile("two_spiral_batch_result.txt");
    for (int j = 0; j < num_hidden_unit; ++j) {
        outfile << input_hidden_v[0][j] << " " << input_hidden_v[1][j] << " " << input_hidden_v[2][j] << endl;
        outfile << input_hidden_u[0][j] << " " << input_hidden_u[1][j] << endl;
    }

    outfile << endl;

    outfile << hidden_output_v[0] << endl;
    for (int j = 0; j < num_hidden_unit; ++j) {
        outfile << hidden_output_v[j + 1] << endl;
        outfile << hidden_output_u[j] << endl;
    }

    std::ifstream testfile("two_spiral_test.txt");

    // Verification
    int total_misclassified = 0;
    int label;
    for (int i = 0; i < num_test_data; ++i) {
        testfile >> input_y[0] >> input_y[1];
        testfile >> label;
        // calculate local of input
        for (int j = 0; j < num_hidden_unit; ++j) {
            input_v[j] = (input_hidden_v[0][j] + input_hidden_v[1][j] * input_y[0] + input_hidden_v[2][j] * input_y[1]);
            input_v[j] += (input_hidden_u[0][j] * input_y[0] * input_y[0] + input_hidden_u[1][j] * input_y[1] * input_y[1]);
            // input of hidden
            hidden_y[j] = 1.0 / (1 + exp(-input_v[j]));
        }

        // calculate local of hidden
        hidden_v = hidden_output_v[0];
        for (int j = 0; j < num_hidden_unit; ++j) {
            hidden_v += (hidden_output_v[j + 1] * hidden_y[j]);
            hidden_v += (hidden_output_u[j] * hidden_y[j] * hidden_y[j]);
        }

        // input of last layer, i.e. output
        output_y = 1.0 / (1 + exp(-hidden_v));

        total_misclassified += ((output_y < 0.5 ? 0 : 1) != label);
        //cout << output_y << " " << label << endl;
    }

    cout << "misclassified points:" << total_misclassified;
}
