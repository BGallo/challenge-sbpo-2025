library(irace)

param_text <- '
initialTemperature "--initialTemperature" i (50, 1000)
coolingRate "--coolingRate" r (0.30, 0.99)
'

writeLines(param_text, "parameters.txt")

parameters <- readParameters("parameters.txt")

target.runner <- function(experiment, scenario) {
  inst_list <- c(
    "datasets/b/instance_0011.txt",
    "datasets/b/instance_0012.txt",
    "datasets/b/instance_0013.txt"
  )

  cfg <- experiment$configuration
  
  results <- sapply(inst_list, function(inst) {
    cmd <- sprintf(
    'java -Xmx4g -jar %s %s output.txt %f %f',
    "/home/pedro/challenge-sbpo-2025/target/ChallengeSBPO2025-1.0.jar",
    inst,
    as.numeric(cfg$initialTemperature),
    as.numeric(cfg$coolingRate)
  )
    
    cat("Running command:", cmd, "\n")
    
    out <- system(cmd, intern = TRUE)
    
    val <- if(length(out) > 0) as.numeric(out[1]) else NA
    
    if (is.na(val)) stop(paste("Java não retornou número válido! Saída:", paste(out, collapse="|")))
    
    val
  })
  
  # IRACE espera uma lista com um elemento chamado 'cost'
  list(cost = -mean(results))  # negativo se maximização
}


scenario <- list(
  maxExperiments = 180,
  targetRunner = target.runner,
  parameters = parameters,
  logFile = "irace.log",
  trainInstancesFile = "instances.txt"
)

irace(scenario)
best <- iraceResults("irace.log")
print(best$allConfigurations) 
print(best$eliteConfigurations)
