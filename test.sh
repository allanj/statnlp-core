mvn clean install
cp  -i /home/roozbeh/Documents/liblbfgs/J_wrapper2_s/build/optimizer.so /home/roozbeh/Downloads/STATNLP_OPTIMIZATION/target
java -jar target/statnlp-core-2017.1-SNAPSHOT.jar         --linearModelClass com.statnlp.example.linear_crf.LinearCRF         --trainPath data/train.data         --testPath data/test.data         --modelPath data/test.model         train test evaluate
#java -classpath /home/roozbeh/Downloads/STATNLP_OPTIMIZATION/target/classes  com.statnlp.example.linear_crf.LinearCRF         --trainPath data/train.data         --testPath data/test.data         --modelPath data/test.model         train test evaluate