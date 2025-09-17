import os
import subprocess

# Caminhos
jar_path = "target/ChallengeSBPO2025-1.0.jar"
input_folder = "datasets/b"
output_folder = "datasets/outputs"

# Cria a pasta de outputs se não existir
os.makedirs(output_folder, exist_ok=True)

# Lista todos os arquivos .txt na pasta de entrada
input_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".txt"))

nThreads = [2,4]

if not input_files:
    print("Nenhum arquivo .txt encontrado na pasta de entrada.")
    exit(1)

# Roda o JAR para cada arquivo
for n in nThreads:
    for i in range(1):
        print(f"\n\n=== Rodando com {n} threads | Iteração {i + 1} ===")
        for filename in input_files:
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_output.txt")

            print(f"\nRodando {input_file} -> {output_file} ...")
            cmd = [
                "java",
                "-Xmx16g",
                "-jar",
                jar_path,
                input_file,
                output_file,
                str(n),
                "67.0",
                "22.0",
                "318",
                "0.88"
            ]

            # Popen para mostrar output em tempo real
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Itera sobre cada linha de output
            for line in proc.stdout:
                print(line, end="")

            proc.wait()

            if proc.returncode != 0:
                print(f"Erro ao rodar {input_file}, return code: {proc.returncode}")
            else:
                print(f"Concluído: {output_file}")
                
        with open("results.log", "a", encoding="utf-8") as f:
            f.write(f"\n")
