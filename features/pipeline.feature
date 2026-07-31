Feature: Log Compression Pipeline
  As an LLM agent
  I want compressed log output
  So that I save tokens when processing log streams

  Scenario: Similar log lines collapse into one cluster
    Given a pipeline with default settings
    When the pipeline processes 3 similar error lines
    Then the output should contain 1 cluster
    And the cluster occurrence count should be 3

  Scenario: Dissimilar lines form separate clusters
    Given a pipeline with default settings
    When the pipeline processes 2 dissimilar lines
    Then the output should contain 2 clusters

  Scenario: Compact format renders bracket counts
    Given a pipeline with compact format
    When the pipeline processes 3 identical lines
    Then the output should contain "[x3]"

  Scenario: TSV format renders tab-separated output
    Given a pipeline with tsv format
    When the pipeline processes 2 identical lines
    Then the output should start with "2\t"

  Scenario: ANSI escapes are stripped during normalization
    Given a pipeline with default settings
    When the pipeline processes 1 ansi-styled line
    Then the output should contain "connection error"
    And the output should not contain escape bytes

  Scenario: Empty input produces empty output
    Given a pipeline with default settings
    When the pipeline processes 0 lines
    Then the output should be empty
