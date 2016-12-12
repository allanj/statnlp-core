DOCUMENTATIONS
==============

Refer to nn_specs.pdf for information regarding the system design.

CODE
====

Source code can be found in the following directory:

src/: Java interface code controlling interactions with NN and CRF
neural_server/: NN code in Torch


INSTALLATION
============

* Install ZeroMQ library: (http://zeromq.org/intro:get-the-software)

git clone https://github.com/zeromq/libzmq
./autogen.sh && ./configure && make -j 4
make check && make install && sudo ldconfig

* Install Torch: (http://torch.ch/docs/getting-started.html#_)

git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc

Type "th" to check if Torch is working.

* Torch dependency libraries

luarocks install nn
luarocks install dkjson
luarocks install lzmq

* Run NN server listening on port 5556 with CPU:

th server.lua -port 5556 -gpuid -1

* Add the following when running the Java program

-Djava.library.path=/usr/local/lib

and add the following in the classpath:

lib/json-20140107.jar
lib/zmq.jar
