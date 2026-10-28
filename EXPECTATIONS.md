# Projekt 1 – Erwartungen

## Aufgabenstellung

In diesem Projekt wird *Abductive Commonsense Reasoning* (ART) untersucht.

Das Modell erhält zwei Beobachtungen sowie zwei mögliche Hypothesen. Ziel ist es, die plausiblere Hypothese auszuwählen.

## Evaluationsmetrik

Als Evaluationsmetrik wird die Accuracy (Genauigkeit) auf dem Testdatensatz verwendet.

Da jeweils zwei Hypothesen zur Auswahl stehen, erreicht ein zufällig entscheidendes Modell erwartungsgemäss ungefähr 50 % Accuracy.

## Modell 1: Word Embeddings

**Erwartete Test-Accuracy: 55–60 %**

Es wird erwartet, dass ein Modell auf Basis von Word Embeddings gewisse semantische Zusammenhänge zwischen den Beobachtungen und Hypothesen erkennen kann.

Bei einer einfachen Zusammenfassung der Wortvektoren, beispielsweise durch Mittelwertbildung, geht allerdings die Information über die Wortreihenfolge weitgehend verloren.

## Modell 2: Word Embeddings + RNN

**Erwartete Test-Accuracy: 60–65 %**

Es wird erwartet, dass ein RNN bessere Ergebnisse erzielen könnte, da es Wörter sequenziell verarbeitet und dadurch Informationen über die Wortreihenfolge und den Kontext berücksichtigen kann.

## Hypothese

Die Hypothese lautet, dass das RNN-basierte Modell eine höhere Accuracy erreicht als das einfachere Word-Embedding-Modell.

Diese Erwartungen wurden vor dem Modelltraining festgelegt. Die angegebenen Werte sind Schätzungen und keine experimentell gemessenen Ergebnisse.