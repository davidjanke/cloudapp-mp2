
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

// >>> Don't Change
public class TitleCount extends Configured implements Tool {
    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new TitleCount(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(this.getConf(), "Title Count");
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setMapperClass(TitleCountMap.class);
        job.setReducerClass(TitleCountReduce.class);

        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setJarByClass(TitleCount.class);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static String readHDFSFile(String path, Configuration conf) throws IOException{
        Path pt=new Path(path);
        FileSystem fs = FileSystem.get(pt.toUri(), conf);
        FSDataInputStream file = fs.open(pt);
        BufferedReader buffIn=new BufferedReader(new InputStreamReader(file));

        StringBuilder everything = new StringBuilder();
        String line;
        while( (line = buffIn.readLine()) != null) {
            everything.append(line);
            everything.append("\n");
        }
        return everything.toString();
    }

// <<< Don't Change

    public static class TitleCountMap extends Mapper<Object, Text, Text, IntWritable> {
        Set<String> stopWords = new HashSet<>();
        String delimiters;
        
        private final static IntWritable ONE = new IntWritable(1);

        @Override
        protected void setup(Context context) throws IOException,InterruptedException {

            Configuration conf = context.getConfiguration();

            String stopWordsPath = conf.get("stopwords");
            String delimitersPath = conf.get("delimiters");

            stopWords = new HashSet<>(Arrays.asList(readHDFSFile(stopWordsPath, conf).split("\n")));
            this.delimiters = readHDFSFile(delimitersPath, conf);
        }


        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        	for(String word : getWordsFromLine(value.toString())) {
        		String normalizedWord = normalizeWord(word);
        		if(!commonWord(normalizedWord)) {
        			context.write(new Text(normalizedWord), ONE);
        		}
        	}
        }
        
        private List<String> getWordsFromLine(String line) {
        	List<String> wordsInLine = new ArrayList<>();
            StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
            
            while(tokenizer.hasMoreElements()) {
            	wordsInLine.add(tokenizer.nextToken());
            }
            return wordsInLine;
        }
        
    	private String normalizeWord(String word) {
    		return word.toLowerCase().trim();
    	}
    	
    	private boolean commonWord(String token) {
    		return stopWords.contains(token);
    	}
        
    }

    public static class TitleCountReduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        	int sum = 0;
        	for(IntWritable count : values) {
        		sum += count.get();
        	}
        	context.write(key, new IntWritable(sum));
        }
    }
}