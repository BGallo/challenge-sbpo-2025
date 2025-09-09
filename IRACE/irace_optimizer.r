library(irace)

param_text <- '
alpha "--alpha" r (0.1, 0.9)
shakeCooldown "--shakeCooldown" i (3, 15)
greedyCooldown "--greedyCooldown" i (3, 15)
k "--k" i (1, 8)
'

writeLines(param_text, "parameters.txt")

parameters <- readParameters("parameters.txt")

instance_ref <- read.table("instances_refs.txt", header = FALSE, stringsAsFactors = FALSE)
colnames(instance_ref) <- c("instance", "ref")
instance_ref_map <- setNames(instance_ref$ref, instance_ref$instance)

target.runner <- function(experiment, scenario) {
  cat("Experimenting with instance: ", experiment$instance, "\n")
  cat("Configuration: ", paste(names(experiment$configuration), experiment$configuration, sep="=", collapse=", "), "\n")
  cfg <- experiment$configuration
  instance <- experiment$instance  # IRACE fornece a instância aqui

  cmd <- sprintf(
    'java -Xmx4g -jar %s %s output.txt %f %d %d %d',
    "D:/Projetos/ProjetosJava/challenge-sbpo-2025/target/ChallengeSBPO2025-1.0.jar",
    instance,
    as.numeric(cfg$alpha),
    as.integer(cfg$shakeCooldown),
    as.integer(cfg$greedyCooldown),
    as.integer(cfg$k)
  )

  out <- system(cmd, intern = TRUE)
  val <- if(length(out) > 0) as.numeric(out[1]) else NA
  if (is.na(val)) stop(paste("Java não retornou número válido! Saída:", paste(out, collapse="|")))

  # normalização
  normalized_val <- val / instance_ref_map[instance]
  cat("Value:", val, "Normalized:", normalized_val, "\n")

  # IRACE espera uma lista com um elemento chamado 'cost'
  list(cost = -normalized_val)  # negativo se maximização
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