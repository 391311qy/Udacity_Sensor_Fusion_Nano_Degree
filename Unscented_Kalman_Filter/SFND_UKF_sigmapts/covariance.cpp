#include <iostream>
#include "Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    VectorXd w(6);
    w<< 1,
        2,
        3,
        4,
        5,
        6;

    MatrixXd X(3,6);
    X << 1,2,3,4,5,6,
         1,2,3,4,5,6,
         1,3,4,5,6,7;

    MatrixXd res;
    res = X*w.asDiagonal()*X.transpose();
    std::cout<<res<<std::endl;
    return 0;
}