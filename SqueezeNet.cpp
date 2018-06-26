#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"





/* TODO:

 * - [ ] create SqueezeNet class

 * - [ ] uniform random tensors (via host->device copy), and CPU initialized tensors

 * - [x] Make `Model` take input and output tensors in forward(), backward()

 * - [ ] Collect total and average times per layer

 * - [ ] implement and benchmark ResNet

 */





void SqueezeNet() {

    TensorDesc input_dim(50, 3, 224, 224);



    Sequential features(input_dim);

    /* features */

    features.addConv(96, 7, 1, 2);
    features.addReLU();
    features.addMaxPool(3, 0, 2);
    
	features.addConv(16, 1, 1, 1);
    features.addReLU();
    features.addConv(64, 1, 1, 1);
    features.addReLU();
    features.addConv(64, 3, 1, 1);
    features.addReLU();
	features.addConv(16, 1, 1, 1);
    features.addReLU();
	features.addConv(64, 1, 1, 1);
    features.addReLU();
	features.addConv(64, 3, 1, 1);
    features.addReLU();
	features.addConv(32, 1, 1, 1);
    features.addReLU();
	features.addConv(128, 1, 1, 1);
    features.addReLU();
	features.addConv(128, 3, 1, 1);
    features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addConv(32, 1, 1, 1);
    features.addReLU();
	features.addConv(128, 1, 1, 1);
    features.addReLU();
	features.addConv(128, 3, 1, 1);
    features.addReLU();
	features.addConv(48, 1, 1, 1);
    features.addReLU();
	features.addConv(192, 1, 1, 1);
    features.addReLU();
	features.addConv(192, 3, 1, 1);
    features.addReLU();
	features.addConv(48, 1, 1, 1);
    features.addReLU();
	features.addConv(192, 1, 1, 1);
    features.addReLU();
	features.addConv(192, 3, 1, 1);
    features.addReLU();
	features.addConv(64, 1, 1, 1);
    features.addReLU();
	features.addConv(256, 1, 1, 1);
    features.addReLU();
	features.addConv(256, 3, 1, 1);
    features.addReLU();
	features.addMaxPool(3, 0, 2);

	features.addConv(64, 1, 1, 1);
    features.addReLU();
	features.addConv(256, 1, 1, 1);
    features.addReLU();
	features.addConv(256, 3, 1, 1);
    features.addReLU();
	//features.addConv(1000, 1, 1, 1);
   // features.addReLU();
	features.addMaxPool(3, 0, 2);




    DEBUG("Dims after Features: " << features.getOutputDesc());



    /* classifier */

    Sequential classifier(features.getOutputDesc());

    // TODO Dropout

    classifier.reshape(input_dim.n, 256 * 14 * 14, 1, 1);
	//15

    // TODO: Dropout

    classifier.addLinear(1000);
	classifier.addReLU();


    Model m(input_dim);

    m.add(features);

    m.add(classifier);



    BenchmarkLogger::new_session("squeeze_net");

    BenchmarkLogger::benchmark(m, 50);

}





int main(int argc, char *argv[])

{

    device_init();



    // enable profiling

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));



    SqueezeNet();



    miopenDestroy(mio::handle());

    return 0;

}
