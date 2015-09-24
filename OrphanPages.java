import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

// >>> Don't Change
public class OrphanPages extends Configured implements Tool {

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new OrphanPages(), args);
        System.exit(res);
    }
// <<< Don't Change

    @Override
    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(this.getConf(), "Orphan Pages");
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(NullWritable.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setMapperClass(LinkCountMap.class);
        job.setReducerClass(OrphanPageReduce.class);

        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setJarByClass(OrphanPages.class);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static class LinkCountMap extends Mapper<Object, Text, IntWritable, IntWritable> {
    	private final static IntWritable ONE = new IntWritable(1);
    	private final static IntWritable ZERO = new IntWritable(0);
    	
    	private static final String ID_DIVIDER = ":";
    	
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        	
        	context.write(new IntWritable(getPageId(value.toString())), ZERO);
        	
        	for(Integer linkedPageId : getLinkedPageIDs(value.toString())) {
       			context.write(new IntWritable(linkedPageId), ONE);
        	}        
        }

		private Integer getPageId(String string) {
			String s = string.split(ID_DIVIDER)[0];
			return Integer.parseInt(s);
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

    public static class OrphanPageReduce extends Reducer<IntWritable, IntWritable, IntWritable, NullWritable> {
    	
        @Override
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        	int sum = 0;
        	for(IntWritable count : values) {
        		sum += count.get();
        	}
        	
        	if(sum == 0) {
        		context.write(key, NullWritable.get());
        	}
        }
    }
}