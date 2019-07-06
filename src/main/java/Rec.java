import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Model;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;
import scala.Tuple3;

import java.io.*;
import java.util.*;

import static java.lang.System.exit;

public class Rec {

    static SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Rec");
    static JavaSparkContext sc = new JavaSparkContext(conf);
    static String FP_addr = "/home/ds/IdeaProjects/DecisionTreeTest/src/main/resources/FP";                //each groupName: longest freqItem
    //static String KMeans_path = "/home/ds/IdeaProjects/DecisionTreeTest/src/main/resources/KMeans/";
    private String data_addr;        //address of training data

    private JavaPairRDD<String, Vector> parsedData;
    private KMeansModel kMeansModel;

    public Rec() {
        this("/home/ds/JobProgram/all-3.txt");
    }

    public Rec(String data_addr) {
        chgTrainingData(data_addr);
    }

    public String getData_addr() {
        return data_addr;
    }

    public void setData_addr(String data_addr) {
        this.data_addr = data_addr;
    }

    public boolean chgTrainingData(String addr) {          //change data content & addr altogether
        setData_addr(addr);
        return chgTrainingData();
    }

    private boolean chgTrainingData() {                  //change data content, same addr
         //clrModel();
         return iniModel();
    }

    private boolean iniModel() {
        kMeans_train();
        return FP_train();
    }

    private void kMeans_train() {
        RDD<Vector> data = preProcess().map(x -> x._2).rdd();
        data.cache();
        // Cluster the data into two classes using KMeans
        int numClusters = 50;
        int numIterations = 100;
        kMeansModel = KMeans.train(data, numClusters, numIterations);

//        Vector[] clusters = kMeansModel.clusterCenters();
//        for (Vector cluster : clusters) {
//            System.out.println(cluster.toString());
//        }
        //exit(3);
    }

    private boolean FP_train() {

        double minSupport = 0.2;
        int numPartitions = 1;

        List<Tuple2<List<String>, List<List<Integer>>>> tmp3 = groups().values().collect();

        Map<String, List<Integer>> toFile = new HashMap<>();

        for(int i=0;i<tmp3.size();i++) {
            String groupName = selName(tmp3.get(i)._1);
            if(i%5==0) {
                int len = tmp3.get(i)._1.size();
                for(int j=0;j<len;j++) {
                    //System.out.println("group"+i+":"+tmp3.get(i)._1.get(j));
                    //System.out.println(tmp3.get(i)._2.get(j));
                }
            }

            System.out.println("group"+i+" : "+groupName);
            JavaRDD<List<Integer>> dataForFP = sc.parallelize(tmp3.get(i)._2);
            FPGrowthModel<Integer> model = new FPGrowth().setMinSupport(minSupport).setNumPartitions(numPartitions).run(dataForFP);

            FPGrowth.FreqItemset<Integer> record = model.freqItemsets().toJavaRDD().mapToPair(x -> new Tuple2<>(x.javaItems().size(), x)).sortByKey(false).take(1).get(0)._2;
            if (toFile.containsKey(groupName)) {                //groupName conflicts
                if(toFile.get(groupName).size() < record.javaItems().size()) {
                    toFile.remove(groupName);
                    toFile.put(groupName, record.javaItems());
                }
            }
            else {
                toFile.put(groupName, record.javaItems());
            }
        }

        File file = new File(FP_addr);
        if(!file.isFile()){
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file, false));
            out.writeObject(toFile);

        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }

    private JavaPairRDD<String, Vector> preProcess() {
        JavaRDD<String> data = sc.textFile(data_addr);
        parsedData = data.mapToPair(s -> {
            String[] sarray = s.split(" ");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length - 1; i++) {
                values[i] = Double.parseDouble(sarray[i + 1]);
            }
            return new Tuple2<>(sarray[0], Vectors.dense(values));
        });
        //parsedData.cache();

        return parsedData;
    }

    private JavaPairRDD<Integer, Tuple2<List<String>, List<List<Integer>>>> groups() {
//        if (parsedData == null)
//            preProcess();
        //JavaPairRDD<String, Vector>
        List<Tuple2<String, Vector>> list = parsedData.collect();
        List<Tuple3<Integer, String, Vector>> list2 = new ArrayList<>();
        for (Tuple2<String, Vector> item : list) {
            list2.add(new Tuple3<>(kMeansModel.predict(item._2), item._1, item._2));
        }

        JavaPairRDD<Integer, Tuple2<List<String>, List<List<Integer>>>> tmp = sc.parallelize(list2).mapToPair(x -> {
            List<String> res1 = new ArrayList<>();
            res1.add(x._2());
            List<List<Integer>> res2 = new ArrayList<>();
            List<Integer> res22 = new ArrayList<>();
            double[] vals = x._3().toArray();
            for(int i=0;i<vals.length;i++) {                  //0,1,1,0....  ->    1,2...
                if(vals[i]==1) {
                    res22.add(i);
                }
            }
            res2.add(res22);

            return new Tuple2<>(x._1(), new Tuple2<>(res1, res2));
        });

        JavaPairRDD<Integer, Tuple2<List<String>, List<List<Integer>>>> tmp2 = tmp.reduceByKey((x, y) -> {
            List<String> res1 = new ArrayList<>(x._1);
            res1.addAll(y._1);
            List<List<Integer>> res2 = new ArrayList<>(x._2);
            res2.addAll(y._2);
            return new Tuple2<>(res1, res2);
        }).sortByKey();

        return tmp2;
    }

//    private List<Integer> format(Vector vector) {
//        double[] items = vector.toArray();
//        List<Integer> res = new ArrayList<>();
//        for(int i=0;i<items.length;i++) {
//            if(items[i]==1) {
//                res.add(i);
//            }
//        }
//        return res;
//    }

    private String selName(List<String> names) {            //whichever appear the most
        JavaRDD<String> tmp0 = sc.parallelize(names);
        JavaPairRDD<String, Integer> tmp1 = tmp0.mapToPair(x -> new Tuple2<>(x, 1));
        JavaPairRDD<Integer, String> tmp2 = tmp1.reduceByKey((x, y) -> x + y).mapToPair(x -> new Tuple2<>(x._2, x._1)).sortByKey(false);
        return tmp2.take(1).get(0)._2;
     }

     public List<String> loadGroupNames() {                               //call from outside
        return new ArrayList<String>(loadRecords().keySet());
     }


    private List<String> jobRec(List<Integer> skills) {   //skills: [3, 6, 2 ,9,0...]
        Map<String, List<Integer>> records = loadRecords();
        List<Tuple2<String, List<Integer>>> list = new ArrayList<>();
        Iterator iter = records.entrySet().iterator();
        while (iter.hasNext()) {
                Map.Entry<String, List<Integer>> entry = (Map.Entry) iter.next();
                list.add(new Tuple2<>(entry.getKey(), entry.getValue()));
        }
        JavaPairRDD<String, List<Integer>> tmp1 = sc.parallelize(list).mapToPair(x -> x);
        JavaPairRDD<Double, String> weightAndName = tmp1.mapToPair(x -> {
            int len1 = x._2.size();
            int len2 = skills.size();
            List<Integer> v1_cpy = new ArrayList<>(x._2);
            v1_cpy.retainAll(x._2);
            int match = v1_cpy.size();
            return new Tuple2<>(match/Math.sqrt(len1*len2), x._1);
        }).sortByKey(false);
        return weightAndName.map(x -> x._2).take(4);
    }

//    private double cosSimilar(final List<Integer> v1, final List<Integer> v2) {
//
//    }

    private Map<String, List<Integer>> loadRecords() {
        File file = new File(FP_addr);
//        if(!file.isFile()) {
//            try {
//                file.createNewFile();
//                iniModel();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
            return (Map<String, List<Integer>>) in.readObject();

        } catch (IOException e) {
            e.printStackTrace();

        } catch (ClassNotFoundException e) {    //???????
            iniModel();
        }
        return null;

    }

    private List<Integer> loadRecord(String job) {
        return loadRecords().get(job);
    }

    private List<Integer> skillRec(String job, List<Integer> skills) {       //jinengbianhao
        if(job==null || job.length()==0) {
                return skillRec(skills);
        }
        List<Integer> item = loadRecord(job);
        List<Integer> res = new ArrayList<>();
        for(int i=0;i<item.size();i++) {
            if(item.get(i)==1 && skills.get(i)==0) {
                res.add(i);
            }
        }
        return res;
    }

    private List<Integer> skillRec(List<Integer> skills) {
        return skillRec(jobRec(skills).get(0), skills);
    }

    public static void main(String[] args) {
        Rec rec = new Rec("/home/ds/JobProgram/all-3 (another copy).txt");
//        Integer[] arr = new Integer[]{7,8,9,10,12,19,25,30,51,52,78, 90, 123, 150};
//        System.out.println(rec.jobRec(Arrays.asList(arr)));
//        List<String> names = rec.loadGroupNames();
//        System.out.println(names);
//        System.out.println(rec.skillRec(names.get(5), Arrays.asList(arr)));

        //System.out.println("kMeans center num:"+rec.kMeansModel.clusterCenters().length);
        //System.out.println(rec.loadRecords());
    }
}
