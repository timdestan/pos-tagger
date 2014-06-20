module Logger
  def log(str)
    $stderr.puts(str) if $DEBUG
  end
end