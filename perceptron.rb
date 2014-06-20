#!/usr/bin/env ruby -wKU

require './viterbi'
require './logger'
require 'set'
require 'json'

class Perceptron
  include Logger
  # Default filename to save to.
  #
  DEFAULT_JSON_FILENAME = "data/saved-perceptron-data.json"
  # Number of passes to make through the training data (T)
  #
  NUM_PASSES = 10
  def initialize()
    @initial_tag_weights = Hash.new(0)  # Hash to get probability of an initial tag.
    @tag_word_weights = Hash.new(0)     # Hash to get the weights for tag/word
    @tag_tag_weights = Hash.new(0)      # Hash to get weights for tag/tag
    @tags = Set.new                     # All tags we have seen.
    @trellis = nil                      # Trellis whose parameters we are tuning.
  end
  
  # Serializes the object to JSON
  #
  def to_json(*a)
    log "Serialization: saving %d tags, %d init weights, %d tag word weights, and %d tag tag weights" %
      [@tags.length, @initial_tag_weights.length, @tag_word_weights.length, @tag_tag_weights.length]
    {
      'json_class' => self.class.name,
      'data' => [ @tags.to_a, @initial_tag_weights, @tag_word_weights, @tag_tag_weights ]
    }.to_json(*a)
  end

  def configure_from_json(tags, initial_tag_weights, tag_word_weights, tag_tag_weights)
    # Tag set
    @tags = tags.to_set
    # initial weights
    initial_tag_weights.each do |k,v|
      @initial_tag_weights[k] = v
    end
    # tag word weights
    tag_word_weights.each do |k,v|
      # JSON bug - assumes all hashes have string keys
      @tag_word_weights[JSON.parse(k)] = v
    end
    # tag tag weights
    tag_tag_weights.each do |k,v|
      # JSON bug - assumes all hashes have string keys
      @tag_tag_weights[JSON.parse(k)] = v
    end
    log "Deserialization: restored %d tags, %d init weights, %d tag word weights, and %d tag tag weights" %
      [@tags.length, @initial_tag_weights.length, @tag_word_weights.length, @tag_tag_weights.length]
  end

  # Create an instance of this class from a serialized JSON object.
  #
  def self.json_create(o)
    rehydrate = new()
    rehydrate.configure_from_json(*o['data'])
    return rehydrate
  end

  def save_to_json(filename=DEFAULT_JSON_FILENAME)
    File.open(filename, "w+") do |file|
      file.write(JSON.generate(self))
    end
  end

  # Train the model using the training examples.
  #
  def train(training_examples)
    # Iterate over the entire training set T times.
    #
    NUM_PASSES.times do |iteration_number|
      log("Beginning iteration number #{iteration_number+1}...")
      training_examples.each do |tagged_example|
        real_tags,words = split_tags_and_words(tagged_example)
        tags = states_of_events(words)
        if (tags != real_tags)
          update_weights(tagged_example, tags)
        end
      end
    end
  end

  # Gets all the possible states.
  #
  def states
    @tags
  end

  # Split tags from words
  # [[DT,the],[ADJ,stupid],[NN,cat]]
  # would split to
  # [[DT,ADJ,NN],[the,stupid,cat]]
  def split_tags_and_words(tagged_example)
    tags = []
    words = []
    tagged_example.each do |tag,word|
      words << word
      tags << tag
      @tags << tag # add it in case we haven't seen this tag yet.
    end
    return tags, words
  end

  # Invalidate memoized data in the trellis.
  #
  def invalidate_trellis
    @trellis = nil
  end

  # Update weights from a wrong guess.
  #
  def update_weights(example, wrong_guess)
    # Can't rely on trellis if we're changing weights.
    invalidate_trellis()
    example.each_index do |index|
      tag,word = example[index]
      wrong_tag = wrong_guess[index]
      if index == 0 
        @initial_tag_weights[tag] += 1
        @initial_tag_weights[wrong_tag] -= 1   
      else
        prev_wrong_tag = wrong_guess[index-1]
        prev_tag = example[index-1][0]
        @tag_tag_weights[[prev_wrong_tag,wrong_tag]] -= 1
        @tag_tag_weights[[prev_tag, tag]] += 1   
      end
      @tag_word_weights[[tag,word]] += 1
      @tag_word_weights[[wrong_tag,word]] -= 1
    end
  end

  # Update running totals (used for averaging)
  # Not in use (SLOOOOOOOOOOOOOOOOW)
  #
  def add_to_total_weights()
    @initial_tag_weights.each do |key,value|
      @total_initial_tag_weights[key] += value
    end
    @tag_tag_weights.each do |key,value|
      @total_tag_tag_weights[key] += value
    end
    @tag_word_weights.each do |key,value|
      @total_tag_word_weights[key] += value
    end
  end

  # "Normalize" total weights, in effect just setting the corresponding
  # weight variable to them. Only used for averaging, so not in use.
  #
  def normalize_total_weights()
    # we don't really care about the magnitudes, so why bother
    # dividing them all by N when it won't make a difference in
    # any calculation?
    #
    @initial_tag_weights = @total_initial_tag_weights
    @tag_tag_weights = @total_tag_tag_weights
    @tag_word_weights = @total_tag_word_weights
  end

  # Gets probability of emitting an event in a given state.
  #
  def get_emission_probability(event, state)
    #log("Returning weight for tag #{state} for word #{event}: #{@tag_word_weights[[state,event]]}")
    @tag_word_weights[[state,event]]
  end

  # Gets probability of transitioning from one state to another.
  #
  def get_transition_probability(src, snk)
    #log("Returning weight for moving from tag #{src} to tag #{snk}: #{@tag_tag_weights[[src,snk]]}")
    @tag_tag_weights[[src,snk]]
  end

  # Gets probability of the model starting in that state.
  #
  def get_start_probability(state)
    #log("Returning weight for starting in tag #{state}: #{@initial_tag_weights[state]}")
    @initial_tag_weights[state]
  end

  def states_of_events(words)
    @trellis = ViterbiTrellis.new(self) unless @trellis
    _,bt = @trellis.prob_and_bt(words)
    return bt
  end
end