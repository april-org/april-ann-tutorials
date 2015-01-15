all: extract_mnist_images extract_mnist_labels

extract_mnist_images: extract_mnist_images.cc
	g++ -Wno-unused-result -O3 -o extract_mnist_images extract_mnist_images.cc

extract_mnist_labels: extract_mnist_labels.cc
	g++ -Wno-unused-result -O3 -o extract_mnist_labels extract_mnist_labels.cc
