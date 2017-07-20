set -o xtrace
mkdir build
jni_path=/usr/lib/jvm/java-8-oracle/include
jni_md_path=/usr/lib/jvm/java-8-oracle/include/linux
javac optimizer.java -d build
javah -d build optimizer 
gcc -I. -L. -I $jni_path -I $jni_md_path -fPIC -shared optimizer.c -l:liblbfgs.so -o build/optimizer.so
java -classpath build optimizer
rm -rf build
