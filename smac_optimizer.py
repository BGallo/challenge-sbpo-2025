import subprocess
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter

# Defina o espaço de parâmetros que quer otimizar
cs = ConfigurationSpace()
cs.add([
    UniformIntegerHyperparameter("antNumber", 5, 50),   # número de formigas
    UniformIntegerHyperparameter("alpha", 1, 5),          # parâmetro alpha
    UniformIntegerHyperparameter("beta", 1, 5),           # parâmetro beta
    UniformFloatHyperparameter("evaporationRate", 0.1, 0.9), # taxa de evaporação
    UniformFloatHyperparameter("epsilon", 0.01, 0.5) # taxa de exploração
])

# Função objetivo: roda seu JAR com parâmetros
def target_function(cfg, seed=0):
    cmd = [
        "java", "-Xmx4g", "-jar", "target/ChallengeSBPO2025-1.0.jar",
        "datasets/a/instance_0001.txt", "output.txt",
        str(cfg["antNumber"]),
        str(cfg["alpha"]),
        str(cfg["beta"]),
        str(cfg["evaporationRate"]),
        str(cfg["epsilon"])
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    
    # Aqui você precisa pegar do output qual é a métrica de desempenho
    # Exemplo: seu JAR imprime "Best Value: X"
    try:
        for line in result.stdout.splitlines():
            if "Best Value" in line:
                value = float(line.split(":")[-1].strip())
                return -value  # SMAC minimiza, então invertemos se for max
    except:
        return 1e6  # penalização se der erro
    
    return 1e6

print("Iniciando otimização com SMAC...")

# Configura o cenário do SMAC
scenario = Scenario(
    cs,
    n_trials=50,  # número de execuções
    output_directory="smac_results",
    deterministic=True
)

# Rodar o otimizador
smac = HyperparameterOptimizationFacade(scenario, target_function)
incumbent = smac.optimize()

print("Melhores parâmetros encontrados:", incumbent)
