package org.sbpo2025.challenge;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class LoggerUtils {

    /**
     * Adiciona valores como novas colunas na primeira linha do arquivo.
     * Se o arquivo não existir, cria e escreve os valores.
     *
     * @param filePath Caminho do arquivo (ex: "resultados.txt")
     * @param values   Valores a adicionar (cada valor será uma coluna)
     */
    public static void appendColumnsToLine(String filePath, Object... values) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            for (Object value : values) {
                writer.write("\t" + value); // adiciona tab e valor
            }
            // NÃO adiciona newLine aqui
        } catch (IOException e) {
            System.err.println("Erro ao escrever no arquivo: " + e.getMessage());
        }
    }
}