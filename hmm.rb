# require('./logex')
require './logger'
require './viterbi'
require 'json'
require 'set'

# Base class for Exceptions raised by this class.
#
class HMMError < Exception
  def initialize(str)
    super(str)
  end
end

# Error raised when someone attempts to add an ID that is
# already in use.
#
class IdInUseError < HMMError
  def initialize(id)
    super("ID #{id} already in use.")
  end
end

class NotImplemented < RuntimeError
end

# Class representing a Hidden Markov Model.
#
class HiddenMarkovModel
  include Logger
  # Default filename to save to.
  #
  DEFAULT_JSON_FILENAME = "data/saved-hmm-data.json"
  # Maximum total of events + states, should be enough.
  #
  MAX_IDS = 1_000_000
  def max_ids
    MAX_IDS
  end
  # Wiggle room for FP comparison.
  #
  EPS = 2.0 ** -24
  # Enum type to identify what an ID refers to.
  #
  class Types
    NONE = nil
    STATE = 1
    # EVENT = 2 -- Decided not to use.
  end
  # Constructor
  #
  def initialize
    @ids = Hash.new(Types::NONE)
    @start = Hash.new(0.0)
    @events = Set.new()
    @transition = {}
    @emission = {}
    @trellis = nil
  end

  # Serializes the object to JSON
  #
  def to_json(*a)
    {
      'json_class' => self.class.name,
      'data' => [ @ids, @events.to_a, @start, @transition, @emission ]
    }.to_json(*a)
  end

  def configure_from_json(ids, events, start, transition, emission)
    ids.each do |k,v|
      @ids[k] = v
    end
    @events = events.to_set
    start.each do |k,v|
      @start[k] = v
    end
    transition.each do |k1,h|
      @transition[k1] = Hash.new(0.0)
      h.each do |k2,v|
        @transition[k1][k2] = v
      end
    end
    emission.each do |k1,h|
      @emission[k1] = Hash.new(0.0)
      h.each do |k2,v|
        @emission[k1][k2] = v
      end
    end
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

  # Record that we've seen this event
  #
  def add_event(id)
    @events << id
  end
  # Check if we've seen some event.
  #
  def has_event?(id)
    @events.include? id
  end

  # Add a state with the given ID
  #
  def add_state(id)
    raise IdInUseError.new(id) if has_id? id
    @ids[id] = Types::STATE
    @transition[id] = Hash.new(0.0)
    @emission[id] = Hash.new(0.0)
    id
  end
  # Enumerates the states.
  #
  def states()
    @ids.select do |id,type|
      # Select only the states.
      type == Types::STATE
    end.map do |id, type|
      # Return only the ids.
      id
    end
  end
  # Is this ID in use anywhere within this HMM?
  #
  def has_id? id
    @ids[id]
  end
  # Ask if we have a state with this ID
  #
  def has_state? id
    @ids[id] == Types::STATE
  end
  # Add a new state with a randomly generated id
  #
  def new_state()
    add_state(new_id())
  end
  # Set probability of emitting event from state.
  #
  def set_emission_probability(event, state, p)
    #log("Setting emission probability of #{event} in #{state} to #{p}")
    add_state(state) unless has_state? state
    add_event(event) unless has_event? event
    @emission[state][event] = p
  end
  # Gets probability of emitting an event in a given state.
  #
  def get_emission_probability(event, state)
    if has_state? state
      @emission[state][event]
    else
      log("unexpected: state #{state} not found.")
      0.0  
    end
  end
  # Set probability of transitioning from
  # src state to snk state.
  #
  def set_transition_probability(src, snk, p)
    #log("Setting transition probability from #{src} to #{snk} = #{p}")
    add_state(src) unless has_state? src
    add_state(snk) unless has_state? snk
    @transition[src][snk] = p
  end
  # Gets probability of transitioning from one state to another.
  #
  def get_transition_probability(src, snk)
    if has_state? src
      @transition[src][snk]
    else
      log("unexpected: source state #{src} not found.")
      0.0
    end
  end
  # Set probability of starting in state.
  #
  def set_start_probability(state, p)
    #log("Setting start probability of #{state} to #{p}")
    add_state(state) unless has_state? state
    @start[state] = p
  end
  # Gets probability of the model starting in that state.
  #
  def get_start_probability(state)
    @start[state]
  end
  # Validate the HMM (make sure all probabilities
  # sum to one when they should).
  #
  def validate()
    validate_start_probabilities()
    validate_exit_probabilities()
    validate_emission_probabilities()
  end
  # Finds the most likely state sequence for a given sequence
  # of events.
  #
  def states_of_events(events)
    @trellis = LogarithmicViterbiTrellis.new(self) unless @trellis
    _,bt = @trellis.prob_and_bt(events)
    return bt
  end
  # Gives log probability of this chain of events.
  #
  def prob_of_events(events)
    @trellis = LogarithmicViterbiTrellis.new(self) unless @trellis
    prob,_ = @trellis.prob_and_bt(events)
    return prob
  end
  private
  def validate_emission_probabilities()
    @emission.each do |state, emissions|
      sum = emissions.values.inject(0) { |acc,v| acc + v }
      if (sum - 1.0).abs > EPS
        log("Emission probabilities for #{state} sum to #{sum}")
      end
    end
  end
  def validate_start_probabilities()
    sum = @start.values.inject(0) { |acc,v| acc + v }
    if (sum - 1.0).abs > EPS
      log("Start probabilities sum to #{sum}")
    end
  end
  def validate_exit_probabilities()
    @transition.each do |src, transitions|
      sum = transitions.values.inject(0) { |acc,v| acc + v }
      if (sum - 1.0).abs > EPS
        log("Exit probabilities for #{src} sum to #{sum}")
      end
    end
  end
  # Generates a new ID for use in this machine.
  #
  def new_id()
    loop do
      id = rand(max_ids)
      return id unless has_id? id
    end
  end
end