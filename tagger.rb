#!/usr/bin/env ruby -wKU

# Author : Tim Destan
# Mailto: tim.destan@liamg.moc
#
# A Laplace-smoothed, bigram, part of speech tagger.

require 'optparse'
require './logger'
autoload :JSON, 'json'
autoload :Set, 'set'
autoload :OpenStruct, 'ostruct'
autoload :HiddenMarkovModel, './hmm'
autoload :Perceptron, './perceptron'

# forms the cartesian product of all enumerables passed
# example usage: cart_prod([1,2,3], [4,5,6])
# yields: [[1,4],[1,5],[1,6],[2,4],....]
# (possibly not in that order, only importance is all the tuples are present)
def cart_prod(*args)
  final_output = [[]]
  until args.empty?
    t, final_output = final_output, []
    b, *args = args
    t.each { |a|
      b.each { |n|
        final_output << a + [n]
      }
    }
  end
  final_output
end

class IO
  # Prints the provided array of pairs to match the format of the
  # corpuses.
  #
  def print_in_corpus_format(ary)
    puts ary.map { |tag,word| "(#{tag} #{word})"}.join(" ")
  end
end

# Given an array, returns an array of the bigrams in
# the provided array.
#
def gen_bigrams(a)
  bigrams = a.zip(a[1..-1])
  bigrams.pop # remove last useless one [lastitem, nil]
  return bigrams
end

class POSTagger
  include Logger
  # Default files to find corpus.
  #
  DEFAULT_TRAINING_CORPUS = "data/f2-21.train.pos"
  DEFAULT_TEST_CORPUS = "data/f2-21.test.pos"
  # Format of a single tagged word within the corpus.
  #
  TAGGED_WORD_RE = /\((\S+)\s+(\S+)\)/

  # Initialize the tagger from a corpus file.
  #
  def initialize(filename, frozen_model=false)
    log "Method in use: #{method2str}"
    if frozen_model
      rehydrate_model_from(filename)
    else
      read_training_corpus(filename)
    end
  end

  # Restore the tagger from the provided file.
  #
  def rehydrate_model_from(filename)
    log("Reading saved #{method2str} model from #{filename}.")
    File.open(filename, "r") do |file|
      @model = JSON.parse(file.read)
    end  
  end

  def save_to_json(filename=default_json_filename)
    @model.save_to_json(filename)
  end

  # Read command line options and run program.
  #
  def self.main(*args)
    options = {
      :training_corpus => DEFAULT_TRAINING_CORPUS,
      :test_corpus => DEFAULT_TEST_CORPUS,
      :debug => $DEBUG,
      :method => "HMM",
      :frozen_tagger => nil
    }
    
    (OptionParser.new do |opts|
      opts.banner = "Usage: #{$0} [options] [-h for help]"
      
      opts.on("-r", "--training-corpus [PATH]", String,
              "Path to file containing training corpus;\n\t" +
              " #{DEFAULT_TRAINING_CORPUS} by default.") do |v|
        options[:training_corpus] = v
      end
      opts.on("-e", "--test-corpus [PATH]", String,
              "Path to file containing test corpus;\n\t" +
              " #{DEFAULT_TEST_CORPUS} by default.") do |v|
        options[:test_corpus] = v
      end
      opts.on("-d", "--[no-]debug", "Set debug mode.") do |v|
        options[:debug] = v
      end
      opts.on("-m", "--method [METHOD]", "Sets classification method.\n\t" +
              " (PERCEPTRON OR HMM). Defaults to HMM.") do |v|
        options[:method] = v
      end
      opts.on("-f", "--tagger-file [FILE]", "Use stored file instead\n\t" +
              "of reading in a training corpus.") do |f|
        options[:frozen_tagger] = f
      end
      opts.on("-h", "--help", "Show this message") do
        puts opts
        exit()
      end
    end).parse!

    # Check for existence of files
    if options[:frozen_tagger].nil? # We need a training corpus
      unless File.exists? options[:training_corpus]
        puts "#{options[:training_corpus]} does not exist."
        exit()
      end
    else
      unless File.exists? options[:frozen_tagger]
        puts "#{options[:frozen_tagger]} does not exist."
        exit()
      end
    end
    unless File.exists? options[:test_corpus]
      puts "#{options[:test_corpus]} is not a directory."
      exit()
    end
    $DEBUG = options[:debug]
    tagger = nil
    case options[:method].upcase
    when "PERCEPTRON"
      if options[:frozen_tagger]
        tagger = PerceptronTagger.new(options[:frozen_tagger], true)
      else
        tagger = PerceptronTagger.new(options[:training_corpus])
        # Save it in case we want to skip this step in the future
        tagger.save_to_json()
      end
    when "HMM"
      if options[:frozen_tagger]
        tagger = HMMTagger.new(options[:frozen_tagger], true)
      else
        tagger = HMMTagger.new(options[:training_corpus])
        # Save it in case we want to skip this step in the future
        tagger.save_to_json()
      end
    else
      puts "#{options[:method]} is not a valid method. Choose HMM or PERCEPTRON."
      exit()
    end
    
    fraction_good = tagger.evaluate(options[:test_corpus])
    puts "Model evaluated %.4f%% of tags correctly." % (fraction_good * 100.0)
  end
end

class HMMTagger < POSTagger
  # Sets the cutoff frequency for classifying as, and
  # token used to represent, unknown or rare words
  # in the corpus.
  #
  UNK_CUTOFF = 5
  UNK_TOKEN = "<UNK>"
  # Reads a single corpus file. Returns a structure containing information
  # about what we read.
  #
  def read_corpus_file(corpus)
    # Initialize all the crap.
    info = OpenStruct.new()
    info.tag_counts = Hash.new(0)
    info.word_counts = Hash.new(0)
    info.tagged_word_counts = Hash.new(0)
    info.tag_pair_counts = Hash.new(0)
    log("Reading corpus #{corpus}...")
    # Read the file.
    File.open(corpus) do |file|
      # Iterate each line.
      file.each_line do |line|
        # Extract the tagged words from the line.
        tagged_words = line.scan(TAGGED_WORD_RE)
        # Record the useful information in our hashes.
        tagged_words.each do |tag,word|
          info.word_counts[word] += 1
          info.tag_counts[tag] += 1
          info.tagged_word_counts[[tag,word]] += 1
        end
        # Make bigrams and note when tags follow one another.
        gen_bigrams(tagged_words).each do |b1,b2|
          info.tag_pair_counts[[b1[0], b2[0]]] += 1
        end
      end
    end
    return info
  end
  # Readable version of the tagging method
  #
  def method2str()
    "Hidden Markov Model"  
  end
  # Default file name for saving model to JSON
  #
  def default_json_filename()
    HiddenMarkovModel::DEFAULT_JSON_FILENAME
  end
  # Idempotent on common words.
  # Replaces rare or unknown words with <unk>.
  #
  def unkify(word)
    unless @word_counts.include? word
      word = UNK_TOKEN
    end 
    return word
  end
  # Precondition the model for Laplacian smoothing based on
  # training information
  #
  def smooth_precondition(info)
    @rare_words = Set.new()
    # Find the rare words.
    info.word_counts.each do |word,frequency|
      if frequency < UNK_CUTOFF
        @rare_words << word
      end
    end
    @word_counts = Hash.new(0)
    # Make a new hash that counts all unk words the same.
    info.word_counts.each do |word, count|
      if @rare_words.include? word 
        @word_counts[UNK_TOKEN] += count
      else
        @word_counts[word] += count
      end
    end
    # Also need to merge all tagged word counts to count
    # unks the same.
    @tagged_word_counts = Hash.new(0)
    info.tagged_word_counts.each do |twp,count|
      tag,word = *twp
      @tagged_word_counts[[tag, unkify(word)]] += count
    end
  end

  # Create the HMM for this data.
  #
  def make_hmm()
    @model = HiddenMarkovModel.new()
    # Set start probabilities.
    total_tag_count = @tag_counts.values.inject(0.0) { |a,x| a + x }
    @tag_counts.each do |tag,count|
      @model.set_start_probability(tag, count / total_tag_count)
    end
    # Set transition probabilities.
    all_tags = @tag_counts.keys
    all_tag_pairs = cart_prod(all_tags, all_tags)
    all_tag_pairs.each do |tag1,tag2|
      @model.set_transition_probability(tag1, tag2,
        (@tag_pair_counts[[tag1,tag2]] + 1.0) /
        (@tag_counts[tag1] + @tag_counts.keys.length))
    end
    # Set emission probabilities.
    @tagged_word_counts.each do |tagged_word, count|
      tag,word = *tagged_word
      @model.set_emission_probability(word, tag, Float(count) / @tag_counts[tag])
    end
    #@model.validate()
  end
  # Reads in the data from a corpus file.
  #
  def read_training_corpus(training_corpus)    
    # Read the info from the provided file.
    training_info = read_corpus_file(training_corpus)
    @tag_counts = training_info.tag_counts
    @tag_pair_counts = training_info.tag_pair_counts
    # Precondition the words for the Laplacian model.
    smooth_precondition(training_info)
    log "Creating Hidden Markov Model..."
    make_hmm()
    log "Successfully created Hidden Markov Model!"
  end
  # Evaluate the generated model against the provided test corpus.
  # Returns the number of correct tags.
  #
  def evaluate(test_corpus)
    log "Evaluating model against #{test_corpus}"
    num_right = 0
    num_possible = 0
    outfile_name = 'data/output-hmm.txt'
    # Read file
    File.open(test_corpus) do |file|
      File.open(outfile_name, 'w+') do |outfile|
        log "Writing results to file #{outfile_name}."
        # Read lines.
        file.readlines.each do |line|
          # Extract tagged words.
          tagged_words = line.scan(TAGGED_WORD_RE)
          # Now just the words.
          just_the_words = tagged_words.map do |tag,word|
            word
          end
          jtw_unk = just_the_words.map do |word|
            if @model.has_event?(word)
              word
            else
              UNK_TOKEN
            end
          end
          states_guess = @model.states_of_events(jtw_unk)
          outfile.print_in_corpus_format(states_guess.zip(just_the_words))
          states_guess.each_with_index do |state,index|
            num_right += 1 if state == tagged_words[index][0]
            num_possible += 1
          end
        end
      end
    end
    log "Tagged #{num_right} correctly out of a possible #{num_possible}."
    return Float(num_right) / num_possible
  end
end

class PerceptronTagger < POSTagger
  # Reads a single corpus file. Returns all the pairs of words.
  #
  def read_corpus_file(corpus)
    return File.open(corpus) do |file|
      file.each_line.map do |line|
        line.scan(TAGGED_WORD_RE)
      end
    end
  end
  # Readable version of the tagging method
  #
  def method2str()
    "Perceptron"  
  end
  # Default file name for saving model to JSON
  #
  def default_json_filename()
    Perceptron::DEFAULT_JSON_FILENAME
  end
  # Reads in the data from a corpus file.
  #
  def read_training_corpus(training_corpus)
    @training_set = read_corpus_file(training_corpus)
    log "Creating and training Perceptron Model"
    train_perceptron()
    log "Successfully trained Perceptron Model" 
  end
  # Create the perceptron for this data and train it
  #
  def train_perceptron()
    @model = Perceptron.new()
    @model.train(@training_set)
  end
  # Evaluate the generated model against the provided test corpus.
  # Returns the number of correct tags.
  #
  def evaluate(test_corpus)
    log "Evaluating model against #{test_corpus}"
    num_right = 0
    num_possible = 0
    outfile_name = "data/output-perceptron.txt"
    # Read file
    File.open(test_corpus) do |file|
      File.open(outfile_name, "w+") do |outfile|
        log "Writing results to file #{outfile_name}."
        # Read lines.
        file.readlines.each do |line|
          # Extract tagged words.
          tagged_words = line.scan(TAGGED_WORD_RE)
          # Now just the words.
          just_the_words = tagged_words.map do |tag,word|
            word
          end
          states_guess = @model.states_of_events(just_the_words)
          outfile.print_in_corpus_format(states_guess.zip(just_the_words))
          states_guess.each_with_index do |state,index|
            num_right += 1 if state == tagged_words[index][0]
            num_possible += 1
          end
        end
      end
    end
    log "Tagged #{num_right} correctly out of a possible #{num_possible}."
    return Float(num_right) / num_possible
  end
end

# Start the script.
POSTagger.main(ARGV) if $0 == __FILE__
