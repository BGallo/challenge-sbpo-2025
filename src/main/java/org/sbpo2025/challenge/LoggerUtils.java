package org.sbpo2025.challenge;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class LoggerUtils {

    /**
     * Adiciona valores como novas colunas na primeira linha do arquivo.
     * Se o arquivo não existir, cria e escreve os valores.
     *
     * @param filePath Caminho do arquivo (ex: "resultados.txt")
     * @param values   Valores a adicionar (cada valor será uma coluna)
     */
    public static void appendColumnsToLine(String filePath, Object... values) {
        try {
            File file = new File(filePath);
            String line = "";

            // Ler a primeira linha, se existir
            if (file.exists()) {
                Scanner scanner = new Scanner(file);
                if (scanner.hasNextLine()) {
                    line = scanner.nextLine();
                }
                scanner.close();
            }

            // Adiciona tabulação se já houver conteúdo
            StringBuilder sb = new StringBuilder(line);
            if (!line.isEmpty()) {
                sb.append("\t");
            }

            // Adiciona os novos valores
            for (int i = 0; i < values.length; i++) {
                sb.append(values[i]);
                if (i < values.length - 1) {
                    sb.append("\t");
                }
            }

            // Reescreve o arquivo com a linha atualizada
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, false))) {
                writer.write(sb.toString());
                writer.newLine(); // mantém o arquivo com uma linha
            }

        } catch (IOException e) {
            System.err.println("Erro ao escrever no arquivo: " + e.getMessage());
        }
    }
}