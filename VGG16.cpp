#include "miopen.hpp"

#include "tensor.hpp"

#include "utils.hpp"

#include "layers.hpp"

#include "multi_layers.hpp"



/* TODO:

 * - [ ] create VGG16 class

 * - [ ] uniform random tensors (via host->device copy), and CPU initialized tensors

 * - [x] Make `Model` take input and output tensors in forward(), backward()

 * - [ ] Collect total and average times per layer

 */



void VGG16net() {

 TensorDesc input_dim(32, 3, 224, 224);


    Sequential features(input_dim);

    /* features */

	features.addConv(64, 3, 1, 1);
	features.addReLU();

	features.addConv(64, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);


	features.addConv(128, 3, 1, 1);
	features.addReLU();

	features.addConv(128, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);


	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addConv(256, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);

	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);

	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();
	
	features.addConv(512, 3, 1, 1);
	features.addReLU();

	features.addMaxPool(2, 0, 2);



    DEBUG("Dims after Features: " << features.getOutputDesc());



    /* classifier */

    Sequential classifier(features.getOutputDesc());

    // TODO Dropout

    classifier.reshape(input_dim.n, 512 * 7 * 7, 1, 1);

    classifier.addLinear(4096);

    classifier.addReLU();

    // TODO: Dropout

    classifier.addLinear(4096);

    classifier.addReLU();

    classifier.addLinear(1000);



    Model m(input_dim);

    m.add(features);

    m.add(classifier);



    BenchmarkLogger::new_session("VGG_16");

    BenchmarkLogger::benchmark(m, 50);

}




int main(int argc, char *argv[])

{

    device_init();

    

    // enable profiling

    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    VGG16net();



    miopenDestroy(mio::handle());

    return 0;
}
