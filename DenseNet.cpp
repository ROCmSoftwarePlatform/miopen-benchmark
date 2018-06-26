#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"




/* TODO:

 * - [ ] create DenseNet class

 * - [ ] uniform random tensors (via host->device copy), and CPU initialized tensors

 * - [x] Make `Model` take input and output tensors in forward(), backward()

 * - [ ] Collect total and average times per layer

 * - [ ] implement and benchmark ResNet

 */





void DenseNet() {


    TensorDesc input_dim(64, 3, 224, 224);



    Sequential features(input_dim);

    /* features */

    features.addConv(16, 3, 1, 1);
    features.addReLU();
    features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(160, 1, 0, 1);
    features.addReLU();
    features.addMaxPool(2, 0, 2);
	//1000

	features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(304, 1, 0, 1);
    features.addReLU();
	features.addMaxPool(2, 0, 2);
	//2030

	features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addConv(12, 3, 1, 1);
    features.addReLU();
	features.addMaxPool(2, 0, 2);




    DEBUG("Dims after Features: " << features.getOutputDesc());



    /* classifier */

    Sequential classifier(features.getOutputDesc());

    // TODO Dropout

    classifier.reshape(input_dim.n, 12 *28 * 28,  1, 1);

    classifier.addLinear(10);

    classifier.addReLU();

  

    Model m(input_dim);

    m.add(features);

    m.add(classifier);



    BenchmarkLogger::new_session("dense_net");

    BenchmarkLogger::benchmark(m, 50);

}





int main(int argc, char *argv[])

{

    device_init();



    // enable profiling

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));



    DenseNet();



    miopenDestroy(mio::handle());

    return 0;

}
