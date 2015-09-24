import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

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

public class PopularityLeague extends Configured implements Tool {

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new PopularityLeague(), args);
        System.exit(res);
    }

    public static class IntArrayWritable extends ArrayWritable {
        public IntArrayWritable() {
            super(IntWritable.class);
        }

        public IntArrayWritable(Integer[] numbers) {
            super(IntWritable.class);
            IntWritable[] ints = new IntWritable[numbers.length];
            for (int i = 0; i < numbers.length; i++) {
                ints[i] = new IntWritable(numbers[i]);
            }
            set(ints);
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = this.getConf();
        FileSystem fs = FileSystem.get(conf);
        Path tmpPath = new Path("/mp2/tmp");
        fs.delete(tmpPath, true);

        Job jobA = Job.getInstance(conf, "Link Count");
        jobA.setOutputKeyClass(IntWritable.class);
        jobA.setOutputValueClass(IntWritable.class);

        jobA.setMapperClass(LinkCountMap.class);
        jobA.setReducerClass(LinkCountReduce.class);

        FileInputFormat.setInputPaths(jobA, new Path(args[0]));
        FileOutputFormat.setOutputPath(jobA, tmpPath);

        jobA.setJarByClass(TopPopularLinks.class);
        jobA.waitForCompletion(true);

        Job jobB = Job.getInstance(conf, "League rank");
        jobB.setOutputKeyClass(IntWritable.class);
        jobB.setOutputValueClass(IntWritable.class);

        jobB.setMapOutputKeyClass(NullWritable.class);
        jobB.setMapOutputValueClass(IntArrayWritable.class);

        jobB.setMapperClass(LeagueMap.class);
        jobB.setReducerClass(LeagueReduce.class);
        jobB.setNumReduceTasks(1);

        FileInputFormat.setInputPaths(jobB, tmpPath);
        FileOutputFormat.setOutputPath(jobB, new Path(args[1]));

        jobB.setInputFormatClass(KeyValueTextInputFormat.class);
        jobB.setOutputFormatClass(TextOutputFormat.class);

        jobB.setJarByClass(PopularityLeague.class);
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
    
    public static Collection<Integer> getLeaugeIds(String leagueIds) {
    	Collection<Integer> ids = new ArrayList<>();
    	for(String idString : leagueIds.split(",")) {
    		ids.add(Integer.parseInt(idString));
    	}
    	return ids;
    }

    public static class LinkCountMap extends Mapper<Object, Text, IntWritable, IntWritable> {
    	private final static IntWritable ONE = new IntWritable(1);
    	
    	private static final String ID_DIVIDER = ":";

    	Set<Integer> league = new HashSet<>();
    	
        @Override
        protected void setup(Context context) throws IOException,InterruptedException {

            Configuration conf = context.getConfiguration();

            String stopWordsPath = conf.get("league");
            
            league.addAll(getLeaugeIds(readHDFSFile(stopWordsPath, conf)));

        }
        
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        	
        	for(Integer linkedPageId : getLinkedPageIDs(value.toString())) {
        		if(league.contains(linkedPageId)) {
        			context.write(new IntWritable(linkedPageId), ONE);
        		}
        	}        
        }

		private Collection<Integer> getLinkedPageIDs(String string) {
			String[] inputSplit = string.split(ID_DIVIDER);
			if(inputSplit.length > 1) {
				return getIdsFromStringList(inputSplit[1]);
			} else {
				return Collections.emptyList();
			}
		}

		private Collection<Integer> getIdsFromStringList(String inputSplit) {
			Collection<Integer> pageIds = new ArrayList<>();
			StringTokenizer tokenizer = new StringTokenizer(inputSplit);			
			while(tokenizer.hasMoreTokens()) {
				pageIds.add(Integer.parseInt(tokenizer.nextToken()));
			}
			return pageIds;
		}
    }

    public static class LinkCountReduce extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
    	@Override
    	protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        	int sum = 0;
        	for(IntWritable count : values) {
        		sum += count.get();
        	}

        	context.write(key, new IntWritable(sum));
    		
    	}
    }
    
    public static class LeagueMap extends Mapper<Text, Text, NullWritable, IntArrayWritable> {
        TreeSet<LinkCount> topLinks = new TreeSet<>();

        
        @Override
        protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        	Integer id = Integer.parseInt(key.toString());
        	Integer count = Integer.parseInt(value.toString());
        	
        	topLinks.add(LinkCount.of(count, id));
        }
        
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
        	for(LinkCount linkCount : topLinks) {
        		context.write(NullWritable.get(), createEmitValue(linkCount));
        	}
        }
        
        private IntArrayWritable createEmitValue(LinkCount linkCount) {
        	Integer[] values = {linkCount.getId(), linkCount.getCount()};
        	return new IntArrayWritable(values);
        }
        
    }

    public static class LeagueReduce extends Reducer<NullWritable, IntArrayWritable, IntWritable, IntWritable> {
        TreeSet<LinkCount> topLinks = new TreeSet<>();
        
        @Override
        protected void reduce(NullWritable key, Iterable<IntArrayWritable> values, Context context) throws IOException, InterruptedException {
        	for(IntArrayWritable linkCountValue : values) {
        		topLinks.add(linkCount(linkCountValue));
        	}
        	
        	for(LinkCount linkCount : topLinks.descendingSet()) {
        		Integer rank = determineRank(linkCount);
        		context.write(new IntWritable(linkCount.getId()), new IntWritable(rank));
        	}
        	
        }

		private Integer determineRank(LinkCount linkCount) {
			int numberOfSmallerElements = topLinks.tailSet(linkCount).size();
			return numberOfSmallerElements;
		}

		private LinkCount linkCount(IntArrayWritable linkCount) {
			Integer[] pair = (Integer[]) linkCount.toArray();
			return LinkCount.of(pair[1], pair[0]);
		}
    }
}

class LinkCount extends Pair<Integer, Integer> {

	public LinkCount(Integer count, Integer id) {
		super(count, id);
	}
	
	public static LinkCount of(Integer count, Integer id) {
		return new LinkCount(count, id);
	}
	
	public Integer getCount() {
		return first;
	}
	
	public Integer getId() {
		return second;
	}
	
	@Override
	public int compareTo(Pair<Integer, Integer> o) {
		return first.compareTo(o.first);
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