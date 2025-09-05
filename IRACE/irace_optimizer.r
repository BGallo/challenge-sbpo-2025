library(irace)

param_text <- '
antNumber "--antNumber" i (5, 50)
alpha "--alpha" r (1.0, 5.0)
beta "--beta" r (1.0, 5.0)
evaporationRate "--evaporationRate" r (0.1, 0.9)
epsilon "--epsilon" r (0.01, 0.5)
'

writeLines(param_text, "parameters.txt")

parameters <- readParameters("parameters.txt")

target.runner <- function(experiment, scenario) {
  inst_list <- c(
    "datasets/a/instance_0001.txt",
    "datasets/a/instance_0002.txt",
    "datasets/a/instance_0020.txt"
  )

  cfg <- experiment$configuration
  
  results <- sapply(inst_list, function(inst) {
    cmd <- sprintf(
      'java -Xmx4g -jar %s %s output.txt %d %f %f %f %f',
      "D:/Projetos/ProjetosJava/challenge-sbpo-2025/target/ChallengeSBPO2025-1.0.jar",
      inst,
      as.integer(cfg$antNumber),
      as.numeric(cfg$alpha),
      as.numeric(cfg$beta),
      as.numeric(cfg$evaporationRate),
      as.numeric(cfg$epsilon)
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
