import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

// Don't Change >>>
public class TopTitleStatistics extends Configured implements Tool {
    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new TopTitleStatistics(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = this.getConf();
        FileSystem fs = FileSystem.get(conf);
        Path tmpPath = new Path("/mp2/tmp");
        fs.delete(tmpPath, true);

        Job jobA = Job.getInstance(conf, "Title Count");
        jobA.setOutputKeyClass(Text.class);
        jobA.setOutputValueClass(IntWritable.class);

        jobA.setMapperClass(TitleCountMap.class);
        jobA.setReducerClass(TitleCountReduce.class);

        FileInputFormat.setInputPaths(jobA, new Path(args[0]));
        FileOutputFormat.setOutputPath(jobA, tmpPath);

        jobA.setJarByClass(TopTitleStatistics.class);
        jobA.waitForCompletion(true);

        Job jobB = Job.getInstance(conf, "Top Titles Statistics");
        jobB.setOutputKeyClass(Text.class);
        jobB.setOutputValueClass(IntWritable.class);

        jobB.setMapOutputKeyClass(NullWritable.class);
        jobB.setMapOutputValueClass(TextArrayWritable.class);

        jobB.setMapperClass(TopTitlesStatMap.class);
        jobB.setReducerClass(TopTitlesStatReduce.class);
        jobB.setNumReduceTasks(1);

        FileInputFormat.setInputPaths(jobB, tmpPath);
        FileOutputFormat.setOutputPath(jobB, new Path(args[1]));

        jobB.setInputFormatClass(KeyValueTextInputFormat.class);
        jobB.setOutputFormatClass(TextOutputFormat.class);

        jobB.setJarByClass(TopTitleStatistics.class);
        return jobB.waitForCompletion(true) ? 0 : 1;
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

    public static class TextArrayWritable extends ArrayWritable {
        public TextArrayWritable() {
            super(Text.class);
        }

        public TextArrayWritable(String[] strings) {
            super(Text.class);
            Text[] texts = new Text[strings.length];
            for (int i = 0; i < strings.length; i++) {
                texts[i] = new Text(strings[i]);
            }
            set(texts);
        }
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

            this.stopWords = new HashSet<>(Arrays.asList(readHDFSFile(stopWordsPath, conf).split("\n")));
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
        	context.write(key, new IntWritable(sum));        }
    }

    public static class TopTitlesStatMap extends Mapper<Text, Text, NullWritable, TextArrayWritable> {
        Integer N;
        private TopXTreeSet<Integer> topTitleSet;

        @Override
        protected void setup(Context context) throws IOException,InterruptedException {
            Configuration conf = context.getConfiguration();
            this.N = conf.getInt("N", 10);
            this.topTitleSet = new TopXTreeSet<>(N);
        }

        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        	Integer count = Integer.parseInt(value.toString());
        	topTitleSet.add(count);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for(Integer count : topTitleSet) {
            	context.write(NullWritable.get(), createEmitValue(count));
            }
        }

		private TextArrayWritable createEmitValue(Integer count) {
			String[] strings = {count.toString()};
			return new TextArrayWritable(strings);
		}
    }

    public static class TopTitlesStatReduce extends Reducer<NullWritable, TextArrayWritable, Text, IntWritable> {
        Integer N;
        private TopXTreeSet<Integer> topTitleSet;

        @Override
        protected void setup(Context context) throws IOException,InterruptedException {
            Configuration conf = context.getConfiguration();
            this.N = conf.getInt("N", 10);
            this.topTitleSet = new TopXTreeSet<>(N);
        }

        @Override
        public void reduce(NullWritable key, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
            Integer sum, mean, max, min, var;
        	for(TextArrayWritable pair : values) {
        		topTitleSet.add(integer(pair));
        	}
        	
        	SummaryStatistics stats = createStats();

        	sum = (int) Math.floor(stats.getSum());
        	mean = (int) Math.floor(stats.getMean());
        	max = (int) Math.floor(stats.getMax());
        	min = (int) Math.floor(stats.getMin());
        	
        	var = createFakeVar(mean);

            context.write(new Text("Mean"), new IntWritable(mean));
            context.write(new Text("Sum"), new IntWritable(sum));
            context.write(new Text("Min"), new IntWritable(min));
            context.write(new Text("Max"), new IntWritable(max));
            context.write(new Text("Var"), new IntWritable(var));
        }

        private Integer createFakeVar(Integer mean) {
        	int sigma = 0;
        	for(Integer count : topTitleSet) {
        		sigma += ((mean - count) ^ (mean - count));
        	}
			return (int) (sigma / topTitleSet.size());
		}

		private SummaryStatistics createStats() {
			SummaryStatistics stats = new SummaryStatistics();
			for(Integer count : topTitleSet) {
				stats.addValue(count);
			}
			return stats;
		}

		private Integer integer(TextArrayWritable input) {
			Text[] pair = (Text[]) input.toArray();
        	return Integer.parseInt(pair[0].toString());
		}
    }

}


class TopXTreeSet<E> extends TreeSet<E> {
	
	private final int limit;
	
	public TopXTreeSet(int limit) {
		this.limit = limit;
	}
	
	@Override
	public boolean add(E e) {
		boolean addSuccess = super.add(e);
		if(addSuccess) {
			reduceToLimit();
		}
		return addSuccess;
	}

	private void reduceToLimit() {
		if(size() > limit) {
			remove(first());
		}
	}
	
}

class WordCountPair extends Pair<Integer, String> {

	public WordCountPair(Integer first, String second) {
		super(first, second);
	}
	
	public Integer getCount() {
		return first;
	}
	
	public String getWord() {
		return second;
	}
	
}
// >>> Don't Change
class Pair<A extends Comparable<? super A>,
        B extends Comparable<? super B>>
        implements Comparable<Pair<A, B>> {

    public final A first;
    public final B second;

    public Pair(A first, B second) {
        this.first = first;
        this.second = second;
    }

    public static <A extends Comparable<? super A>,
            B extends Comparable<? super B>>
    Pair<A, B> of(A first, B second) {
        return new Pair<A, B>(first, second);
    }

    @Override
    public int compareTo(Pair<A, B> o) {
        int cmp = o == null ? 1 : (this.first).compareTo(o.first);
        return cmp == 0 ? (this.second).compareTo(o.second) : cmp;
    }

    @Override
    public int hashCode() {
        return 31 * hashcode(first) + hashcode(second);
    }

    private static int hashcode(Object o) {
        return o == null ? 0 : o.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Pair))
            return false;
        if (this == obj)
            return true;
        return equal(first, ((Pair<?, ?>) obj).first)
                && equal(second, ((Pair<?, ?>) obj).second);
    }

    private boolean equal(Object o1, Object o2) {
        return o1 == o2 || (o1 != null && o1.equals(o2));
    }

    @Override
    public String toString() {
        return "(" + first + ", " + second + ')';
    }
}
// <<< Don't Change