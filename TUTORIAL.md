# Costruire un'applicazione agentica in Rust su Amazon Bedrock, osservata con Langfuse

> Percorso di apprendimento passo-passo, pensato per chi parte da un'idea rudimentale di cosa sia un harness per LLM.
> Ogni passo è una casella: segna `[x]` quando è fatto. Stato dell'arte: settembre 2026.

---

## 0. Come usare questo tutorial

### 0.1 Struttura

- I moduli vanno in ordine. I moduli **1–9** sono il cuore (concetti, Bedrock, Langfuse, harness, prompt, contesto, pipeline). I moduli **10–15** sono la parte "production" (router, valutazione, test, sicurezza, osservabilità, deploy). Il modulo **16** raccoglie approfondimenti opzionali.
- Ogni modulo ha sempre le stesse sezioni: **Obiettivo**, **Perché conta**, **Passi** (le caselle da spuntare), **Fatto quando** (criterio oggettivo di completamento), **Autoverifica** (cosa devi saper spiegare a voce), **Approfondimenti**.
- Legenda: ⏱ stima di tempo · 🧪 esperimento da annotare nel Diario (Appendice E) · ⚠️ attenzione · 💡 concetto chiave · 📚 lettura consigliata · 🔭 passo specifico su Langfuse.
- Ogni decisione non banale va scritta come ADR (Architecture Decision Record, Appendice C). Ogni esperimento con numeri va nel Diario. Sono due abitudini che valgono più di qualsiasi strumento.

### 0.2 Stato di avanzamento

- [ ] Modulo 1 · Fondamenti: LLM, harness, agenti
- [ ] Modulo 2 · Setup ambiente e repository Rust
- [ ] Modulo 3 · Attivare Amazon Bedrock
- [ ] Modulo 4 · Prima chiamata a Bedrock da Rust
- [ ] Modulo 5 · Attivare Langfuse e vedere le chiamate
- [ ] Modulo 6 · Prompt engineering sistematico con Langfuse Prompt Management
- [ ] Modulo 7 · Tool calling e primo agent loop
- [ ] Modulo 8 · Gestione del contesto (context engineering)
- [ ] Modulo 9 · La pipeline del progetto guida
- [ ] Modulo 10 · Router di modelli
- [ ] Modulo 11 · Valutare i modelli con dataset ed esperimenti
- [ ] Modulo 12 · Suite di test anti-regressione
- [ ] Modulo 13 · Sicurezza, privacy e guardrail
- [ ] Modulo 14 · Osservabilità in produzione con Langfuse
- [ ] Modulo 15 · Deploy in produzione
- [ ] Modulo 16 · Approfondimenti opzionali

### 0.3 Il progetto guida

Per non imparare "a vuoto", tutti i moduli costruiscono un'unica applicazione che cresce passo dopo passo. È ispirata al documento interno *AI Prompt Processing Pipeline* (Notion), preso come spunto e non come specifica: la sezione sui modelli di quel documento è datata e va ignorata.

**Un assistente conversazionale per un servizio sanitario**, che per ogni messaggio dell'utente esegue:

1. `STATE_READ` · legge stato conversazione e profilo utente dal DB (nessuna chiamata LLM).
2. `SAFETY` · classificatore LLM che fa da cancello: se rileva un rischio acuto, risponde con un messaggio di escalation e si ferma.
3. `RETRIEVAL` · ricerca su una knowledge base (vettoriale, nessuna chiamata LLM).
4. `GENERATION` · chiamata LLM principale con **output strutturato**: testo di risposta più segnali (stato affettivo, argomenti, riferimenti alla KB, hint per il profilo).
5. `GUARDRAILS` · seconda chiamata LLM che valida la risposta; se fallisce, `GENERATION` viene rieseguita con istruzioni correttive.
6. `STATE_WRITE` · aggiornamento sincrono dello stato conversazione.
7. `PROFILE_UPDATE` · chiamata LLM **asincrona** che aggiorna il profilo utente dopo la consegna della risposta.

Vedrai comparire in questa pipeline quasi tutti i temi del tutorial: prompt diversi per step diversi, modelli diversi per step diversi (router), gestione del contesto (finestra di turni a budget di token, profilo compatto), guardrail, streaming con validazione in parallelo, aggiornamenti atomici, una traccia Langfuse per ogni messaggio con uno span per step, test per step.

⚠️ Il dominio sanitario è reale ma qui è un **playground**: usa solo dati sintetici, mai dati di pazienti veri.

### 0.4 Decisioni di partenza (assunzioni esplicite)

| Tema | Decisione | Perché |
|---|---|---|
| Linguaggio | Rust (edition 2024), runtime `tokio` | Richiesta esplicita. Ottimo per capire l'harness "a mano": niente magia. |
| Provider | Amazon Bedrock tramite **Converse API** con l'AWS SDK for Rust ufficiale | Converse è l'API unificata di Bedrock: stessi tipi per tutti i modelli, tool use, streaming, caching, structured output. |
| Regione | Un'area **EU** (`eu-west-1` o `eu-central-1`) e inference profile `eu.*` | Residenza dei dati (GDPR). Vedi Modulo 3. |
| Modelli | Famiglia Claude su Bedrock come base; almeno un modello non-Anthropic (Amazon Nova, Llama, Mistral) per imparare a confrontare e a instradare | Il router e la valutazione hanno senso solo con alternative reali. |
| Piattaforma LLM engineering | **Langfuse**, usato in tutte e quattro le sue aree: tracing, prompt management, evaluation (dataset, esperimenti, LLM-as-a-judge, annotazioni), dashboard e alert | È open source, ha una regione Cloud in UE e si può self-hostare; si integra da Rust via OpenTelemetry e API pubblica. Vedi Modulo 5. |
| Framework agentici | **Nessuno** nel core: l'harness si scrive a mano. Rig (il framework Rust più maturo) è un confronto opzionale nel Modulo 16 | Scrivere il loop una volta è il modo più rapido per capire cosa fanno i framework. |
| Persistenza | SQLite in locale (`rusqlite` o `sqlx`), PostgreSQL con `pgvector` quando serve il vettoriale | Semplice da avviare, realistico per la produzione. |

---

## Modulo 1 · Fondamenti: LLM, harness, agenti

⏱ 3–4 ore · solo lettura e appunti, niente codice.

**Obiettivo.** Avere un vocabolario preciso prima di scrivere una riga di codice.

**Perché conta.** Il 70% degli errori nelle applicazioni agentiche nasce da un modello mentale sbagliato di cosa fa (e non fa) un LLM: si scrive un prompt "che funziona" senza sapere perché, e poi non si riesce a fare debugging.

### Passi

- [ ] 1.1 💡 **Cos'è un LLM per chi lo integra.** Una funzione `(sequenza di token) → (distribuzione sul prossimo token)`, campionata in loop. Non ha memoria tra una chiamata e l'altra: tutto quello che "sa" della conversazione glielo rimandi tu ogni volta. Scrivi questa frase con parole tue nel Diario.
- [ ] 1.2 💡 **Token, context window, costo.** I token sono l'unità di misura di tutto: costo (prezzo per milione di token in ingresso e in uscita), limiti (context window), latenza (i token in uscita si pagano in tempo). Un testo italiano vale circa 1 token ogni 3–4 caratteri. La context window dei modelli Claude 5 su Bedrock è 1M token, ma "entra" non vuol dire "funziona bene": la qualità degrada quando riempi la finestra di rumore.
- [ ] 1.3 💡 **I ruoli dei messaggi.** `system` (le istruzioni dell'operatore: chi sei, cosa fai, in che formato rispondi), `user`, `assistant`. Il system prompt è la leva più potente che hai.
- [ ] 1.4 💡 **Parametri di inferenza.** `max_tokens` (tetto sull'output), `temperature` (0 = più deterministico, non "deterministico"), `top_p`, `stop_sequences`. Per i modelli Claude più recenti conta anche il **thinking adattivo** e il livello di **effort** (quanto il modello "ragiona" prima di rispondere). Nota: su Claude 5 `temperature` e `top_p` non sono più accettati; si controlla la profondità con `effort`.
- [ ] 1.5 💡 **Tool calling (function calling).** Il modello non esegue niente: emette un blocco `tool_use` con nome e argomenti JSON, tu esegui la funzione e rimandi un blocco `tool_result`. Il loop `chiamata → tool_use → esegui → tool_result → chiamata…` è **l'harness**. Un "agente" è un harness in cui il modello decide quali tool usare e quando fermarsi.
- [ ] 1.6 💡 **Structured output.** Chiedere al modello di rispondere in un JSON che rispetta uno schema. Si ottiene in due modi: forzando un tool con lo schema voluto, oppure con la funzione nativa di structured output (su Bedrock: `outputConfig.textFormat` con `json_schema`). Serve ovunque nel progetto guida.
- [ ] 1.7 💡 **Streaming.** Ricevere i token man mano invece che tutti alla fine. Cambia la latenza percepita (time-to-first-token) e complica il codice (eventi parziali, tool use a pezzi).
- [ ] 1.8 📚 **Workflow vs agenti.** Leggi *Building effective agents* (Anthropic). Impara a distinguere: **workflow** (il codice decide il flusso, l'LLM fa i singoli step) e **agente** (l'LLM decide il flusso). Il progetto guida è un workflow con pezzi agentici: è la forma giusta per la maggior parte dei prodotti.
- [ ] 1.9 📚 **I cinque pattern di composizione.** Prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer. Per ognuno scrivi nel Diario una riga: "nel progetto guida lo userei per…". (Suggerimento: `SAFETY` è routing, `GUARDRAILS` è evaluator-optimizer.)
- [ ] 1.10 💡 **Quando NON serve un agente.** Quattro domande: il task è multi-step e difficile da specificare a priori? Il valore giustifica costo e latenza extra? Il modello è capace su questo tipo di task? Gli errori sono recuperabili? Se una risposta è "no", resta su chiamata singola o workflow.
- [ ] 1.11 💡 **Non determinismo.** Anche a `temperature = 0` due chiamate identiche possono dare output diversi. Questo cambia tutto nei test (Modulo 12): non si testa "l'output è X", si testa "l'output rispetta le proprietà P, su N ripetizioni, oltre una soglia".
- [ ] 1.12 💡 **Osservare un sistema LLM.** Tre parole che useremo sempre: **trace** (tutto ciò che succede per una richiesta), **observation** (un pezzo della trace: uno span generico, una *generation* cioè una chiamata al modello, un evento), **score** (un numero o un'etichetta attaccati a una trace o a un'observation: qualità, feedback, esito guardrail). Sono il modello dati di Langfuse e di quasi tutti gli strumenti del settore.
- [ ] 1.13 Compila il glossario (Appendice A) con le tue parole. Se una definizione non ti viene, non hai ancora capito il concetto.

**Fatto quando** hai scritto nel Diario le definizioni di: token, context window, system prompt, tool use, harness, agente, workflow, structured output, streaming, effort, trace, observation, score.

**Autoverifica.** Spiega a voce, in due minuti, perché "l'LLM ricorda la conversazione" è falso e cosa fa davvero il tuo codice per dare quell'illusione.

**Approfondimenti.** *Effective context engineering for AI agents* e *Writing effective tools for agents* (Anthropic Engineering): li riprenderemo nei Moduli 7 e 8.

---

## Modulo 2 · Setup ambiente e repository Rust

⏱ 2–3 ore.

**Obiettivo.** Un workspace Cargo con la struttura definitiva, CI minima, lint e test che girano a vuoto.

**Perché conta.** Separare fin da subito "cosa parla con il provider", "cosa decide il flusso" e "cosa osserva" è ciò che renderà possibili mock, test, router e tracing nei moduli successivi.

### Passi

- [ ] 2.1 Verifica la toolchain: `rustup update stable`, `cargo --version` (qui risulta già installato cargo 1.97). Installa `cargo install cargo-nextest cargo-watch cargo-lambda just`. Valuta `cargo-insta` (snapshot test) e `cargo-deny` (audit dipendenze).
- [ ] 2.2 Crea il workspace. Struttura suggerita (una crate per responsabilità, così i confini restano netti):

  ```text
  agentic-app-playground/
  ├── Cargo.toml                 # [workspace]
  ├── justfile                   # comandi ricorrenti (test, eval, run)
  ├── crates/
  │   ├── llm-core/              # tipi neutri: Message, ContentBlock, ToolSpec, LlmRequest/Response, trait LlmClient
  │   ├── llm-bedrock/           # implementazione LlmClient su aws-sdk-bedrockruntime (Converse)
  │   ├── observability/         # init tracing + OpenTelemetry, exporter Langfuse, helper per attributi, client API Langfuse
  │   ├── prompts/               # caricamento prompt: da Langfuse (con cache) con fallback ai file in repo
  │   ├── harness/               # agent loop, tool registry, gestione contesto, pipeline steps
  │   ├── router/                # ModelRegistry, ModelRouter, fallback, circuit breaker
  │   ├── evals/                 # runner di valutazione: dataset, grader, esperimenti su Langfuse, report
  │   └── app/                   # binario: CLI (clap) e server HTTP (axum) con streaming SSE
  ├── prompts/                   # snapshot dei prompt (fallback e diff in PR), uno per step
  ├── datasets/                  # casi di test e golden set (JSONL), solo dati sintetici
  ├── docs/adr/                  # Architecture Decision Records
  └── TUTORIAL.md
  ```

- [ ] 2.3 Dipendenze di base nel workspace: `tokio` (full), `serde`, `serde_json`, `schemars` (genera JSON Schema dalle struct: lo userai per tool e structured output), `thiserror`, `anyhow`, `tracing`, `tracing-subscriber` (env-filter, json), `dotenvy`, `clap`, `async-trait`, `futures`. Le versioni le prendi da crates.io il giorno in cui installi: non fidarti di numeri letti in tutorial.
- [ ] 2.4 Configura `.env` (mai committato) e `.env.example` (committato) con `AWS_PROFILE`, `AWS_REGION`, `BEDROCK_MODEL_FAST`, `BEDROCK_MODEL_MAIN`, e già i segnaposto `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (li compili nel Modulo 5). Aggiungi `.env` a `.gitignore`.
- [ ] 2.5 Lint e formato: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`. Un `justfile` con `just check`, `just test`, `just eval`, `just run`.
- [ ] 2.6 CI su GitHub Actions: un workflow `ci.yml` che su ogni PR esegue fmt, clippy e `cargo nextest run` **senza credenziali** (i test che parlano con Bedrock o Langfuse saranno `#[ignore]`, Modulo 12). Un secondo workflow `nightly.yml`, vuoto per ora, che poi eseguirà contract test ed esperimenti.
- [ ] 2.7 Scrivi `docs/adr/0001-struttura-workspace.md` usando il template dell'Appendice C. È il tuo primo ADR: breve, anche cinque righe.
- [ ] 2.8 Primo commit (in inglese, imperativo: `Add workspace skeleton and CI`).

**Fatto quando** `just check` e `just test` passano su un workspace vuoto, e la CI è verde.

**Autoverifica.** Perché `llm-core` non deve dipendere da `aws-sdk-*` né da `observability`? (Per poter mockare il provider nei test, aggiungere un secondo provider senza toccare l'harness, e cambiare backend di osservabilità senza toccare la logica.)

---

## Modulo 3 · Attivare Amazon Bedrock

⏱ 2–3 ore, di cui molta attesa su console AWS.

**Obiettivo.** Un account AWS con Bedrock attivo in EU, credenziali a privilegio minimo, modelli abilitati, budget sotto controllo e una prima chiamata riuscita dalla CLI.

**Perché conta.** Bedrock ha un modello di accesso diverso da un provider "API key": IAM, regioni, abilitazione modelli, quote. Sbagliare qui produce errori criptici (`AccessDeniedException`, `ValidationException` su model id) che sembrano bug del codice.

### Passi

- [ ] 3.1 **Account.** Usa un account AWS **sandbox** dedicato (se l'azienda ha AWS Organizations, chiedine uno; altrimenti un account personale). MFA sull'utente root, root mai usato per lavorare.
- [ ] 3.2 **Identità.** Preferisci IAM Identity Center (SSO) con un permission set; in alternativa un utente IAM con MFA e access key a rotazione. Per iniziare la policy gestita `AmazonBedrockFullAccess` va bene; **prima del Modulo 15** la sostituisci con una policy che concede solo `bedrock:InvokeModel` e `bedrock:InvokeModelWithResponseStream` (sono le azioni usate anche da Converse e ConverseStream) sugli ARN degli inference profile che usi davvero.
- [ ] 3.3 **AWS CLI.** È già installata (v2.36). Configura un profilo: `aws configure sso` oppure `aws configure --profile bedrock-playground`. Verifica con `aws sts get-caller-identity --profile bedrock-playground`.
- [ ] 3.4 **Regione.** Scegli `eu-west-1` (Irlanda) o `eu-central-1` (Francoforte) come regione "di casa". ⚠️ Milano (`eu-south-1`) ha un catalogo modelli più povero. 💡 Con la **cross-region inference** (inference profile con prefisso `eu.`) Bedrock instrada tra le regioni EU restando nell'area geografica: più capacità, dati che non lasciano l'UE. Il prefisso `global.` invece può uscire dall'UE: non usarlo in questo progetto.
- [ ] 3.5 **Abilitazione modelli.** Console → Bedrock → *Model catalog*. Per i modelli **Anthropic** c'è un modulo *use case* da compilare una sola volta per account: l'accesso è immediato dopo l'invio. Abilita anche **Amazon Nova** (nessun modulo) e almeno un modello open-weight (Llama o Mistral) per il Modulo 11. L'abbonamento AWS Marketplace viene creato automaticamente alla prima invocazione se l'identità ha i permessi.
- [ ] 3.6 **Trova gli ID giusti.** Gli ID cambiano: non copiarli dai tutorial, leggili dal tuo account.

  ```bash
  aws bedrock list-foundation-models --region eu-west-1 --by-provider anthropic --query 'modelSummaries[].modelId'
  aws bedrock list-inference-profiles --region eu-west-1 --query 'inferenceProfileSummaries[].inferenceProfileId'
  ```

  Esempi di formato (da verificare): `eu.anthropic.claude-sonnet-5`, `eu.anthropic.claude-opus-5`, `eu.anthropic.claude-haiku-4-5-20251001-v1:0`. Segna in `.env` un modello **veloce ed economico** (`BEDROCK_MODEL_FAST`, tipicamente Haiku) e uno **principale** (`BEDROCK_MODEL_MAIN`, tipicamente Sonnet).
- [ ] 3.7 **Quote.** Console → Service Quotas → Bedrock: cerca *requests per minute* e *tokens per minute* per i modelli scelti. Annotale: sono il tetto che il router (Modulo 10) dovrà rispettare. Richiedi un aumento solo se serve.
- [ ] 3.8 **Budget.** AWS Budgets: un budget mensile (per esempio 30 €) con alert al 50% e all'80% via email. Attiva anche *Cost allocation tags* con un tag `project=agentic-playground`: lo userai per riconciliare i costi calcolati da Langfuse con la bolletta (Modulo 14).
- [ ] 3.9 **Logging delle invocazioni.** Console → Bedrock → *Settings* → *Model invocation logging* verso CloudWatch Logs. Utile nei primissimi giorni di debugging; da quando avrai Langfuse (Modulo 5) diventa ridondante. ⚠️ I log contengono i prompt completi: in produzione con dati personali questa scelta va rivista (Modulo 13).
- [ ] 3.10 🧪 **Prima chiamata dalla CLI.**

  ```bash
  aws bedrock-runtime converse \
    --region eu-west-1 --profile bedrock-playground \
    --model-id "$BEDROCK_MODEL_FAST" \
    --messages '[{"role":"user","content":[{"text":"Rispondi con una sola parola: pronto?"}]}]' \
    --inference-config '{"maxTokens":50}'
  ```

  Guarda la risposta: `output.message.content`, `stopReason`, `usage.inputTokens`, `usage.outputTokens`, `metrics.latencyMs`. Annota nel Diario i token e la latenza: è la tua prima misura.
- [ ] 3.11 📚 **Prezzi.** Leggi la pagina prezzi di Bedrock per i modelli scelti. Calcola a mano quanto costa la chiamata appena fatta. Impara la differenza tra token in ingresso, in uscita, **cache write** e **cache read** (le letture da cache costano circa il 10% dell'input normale). Questi prezzi li inserirai in Langfuse nel Modulo 5.
- [ ] 3.12 💡 **Nota su "Claude Platform on AWS".** Dal 2026 Anthropic offre anche un accesso gestito direttamente da Anthropic dentro AWS (autenticazione SigV4, fatturazione Marketplace) con parità di funzionalità con l'API Anthropic e ID modello senza prefisso. È un'alternativa a Bedrock, non la stessa cosa: Bedrock è gestito da AWS, ha il proprio catalogo multi-vendor, Guardrails, Knowledge Bases, AgentCore. In questo tutorial restiamo su Bedrock perché l'obiettivo è imparare il provider; ne riparliamo nel Modulo 16.

**Fatto quando** la chiamata 3.10 risponde, il budget è attivo, e in `.env` hai due model id validi della regione EU.

**Autoverifica.** Cosa succede se chiami un modello con un model id `us.*` da `eu-west-1`? E perché un inference profile `eu.*` è preferibile a un model id "nudo" senza prefisso?

---

## Modulo 4 · Prima chiamata a Bedrock da Rust

⏱ 4–6 ore.

**Obiettivo.** Una crate `llm-bedrock` che implementa il trait `LlmClient` definito in `llm-core`, con chiamata semplice, streaming, gestione errori e misura di token e latenza.

**Perché conta.** Qui nasce il confine più importante dell'architettura: l'harness parlerà solo con `LlmClient`, mai con l'SDK.

### Passi

- [ ] 4.1 In `llm-core` definisci i tipi neutri. Non copiare i tipi dell'SDK: modella solo ciò che ti serve.

  ```rust
  pub enum Role { User, Assistant }
  pub enum ContentBlock { Text(String), ToolUse { id: String, name: String, input: serde_json::Value },
                          ToolResult { tool_use_id: String, content: String, is_error: bool }, CachePoint }
  pub struct Message { pub role: Role, pub content: Vec<ContentBlock> }
  pub struct LlmRequest { pub model: ModelId, pub system: Vec<SystemBlock>, pub messages: Vec<Message>,
                          pub tools: Vec<ToolSpec>, pub max_tokens: u32, pub output_schema: Option<schemars::Schema>,
                          pub extra: serde_json::Value /* campi specifici del modello, es. effort */ }
  pub struct LlmResponse { pub message: Message, pub stop_reason: StopReason, pub usage: Usage, pub latency: Duration }
  pub struct Usage { pub input_tokens: u32, pub output_tokens: u32, pub cache_read_tokens: u32, pub cache_write_tokens: u32 }

  #[async_trait::async_trait]
  pub trait LlmClient: Send + Sync {
      async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError>;
      async fn stream(&self, req: LlmRequest) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError>;
  }
  ```

- [ ] 4.2 In `llm-bedrock` aggiungi `aws-config` (feature `behavior-version-latest`) e `aws-sdk-bedrockruntime`. Costruisci il client una volta sola (è costoso) e condividilo con `Arc`.

  ```rust
  let cfg = aws_config::defaults(BehaviorVersion::latest()).region(Region::new("eu-west-1")).load().await;
  let client = aws_sdk_bedrockruntime::Client::new(&cfg);
  ```

- [ ] 4.3 Implementa `complete` con `client.converse()`: `model_id`, `system(SystemContentBlock::Text(..))`, `messages(..)`, `inference_config(InferenceConfiguration::builder().max_tokens(..))`. Mappa la risposta: `output()` → `ConverseOutput::Message`, `stop_reason()`, `usage()`, `metrics().latency_ms()`. 📚 Consulta docs.rs di `aws_sdk_bedrockruntime` per i nomi esatti: i builder dell'SDK cambiano tra versioni.
- [ ] 4.4 **Errori.** Mappa le eccezioni del servizio in un `LlmError` con varianti che contano per il chiamante: `Throttled` (ritentabile), `ModelNotReady`/`ServiceUnavailable` (ritentabile), `ValidationError` (bug tuo, non ritentare), `AccessDenied` (config), `ContextTooLong`, `Other`. 💡 L'SDK ha già retry con backoff esponenziale e jitter: configuralo (`RetryConfig`) invece di riscriverlo; il router (Modulo 10) aggiungerà solo il fallback tra modelli.
- [ ] 4.5 **Streaming.** Implementa `stream` con `converse_stream()`: ricevi `ConverseStreamOutput` con eventi `MessageStart`, `ContentBlockStart`, `ContentBlockDelta` (testo o pezzi di JSON del tool), `ContentBlockStop`, `MessageStop`, `Metadata` (usage). Traducili in un tuo `LlmEvent`. ⚠️ Gli argomenti dei tool arrivano a pezzi come stringa JSON: accumula e fai il parse solo a `ContentBlockStop`.
- [ ] 4.6 **Parametri specifici del modello.** Converse è neutro; ciò che è specifico di un vendor passa in `additional_model_request_fields` (un `Document`). Per Claude ci passano, ad esempio, `thinking` e `output_config.effort`. Fai in modo che `LlmRequest.extra` finisca lì.
- [ ] 4.7 **Prompt caching.** Converse supporta il blocco `cachePoint`: marca la fine del prefisso stabile (system prompt, definizioni tool). Aggiungi `ContentBlock::CachePoint` e verifica che `usage.cache_read_tokens` diventi maggiore di zero alla seconda chiamata identica. Se resta zero, qualcosa nel prefisso cambia a ogni chiamata (timestamp, ordine dei campi).
- [ ] 4.8 🧪 Un binario `app hello` che fa una chiamata e stampa: risposta, stop reason, token, latenza, costo stimato (una tabella prezzi hardcoded per ora). Ripeti 5 volte la stessa chiamata: annota la varianza di latenza e la presenza della cache.
- [ ] 4.9 🧪 Streaming da CLI: stampa i token man mano che arrivano. Misura il **time-to-first-token** e il tempo totale. Confrontali con la chiamata non-streaming.
- [ ] 4.10 Aggiungi `tracing`: uno `span` per chiamata con campi `model`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `latency_ms`, `stop_reason`. Nel Modulo 5 questi span diventeranno *generation* in Langfuse senza toccare `llm-bedrock`.
- [ ] 4.11 Un secondo `LlmClient` finto, `FakeLlmClient`, in `llm-core` (dietro feature `test-util`): risponde con risposte preconfigurate. Ti serve dal Modulo 7 in poi per testare l'harness senza rete.

**Fatto quando** `app hello` e `app hello --stream` funzionano, gli errori di throttling e di validazione sono distinguibili nei log, e `cache_read_tokens > 0` alla seconda chiamata.

**Autoverifica.** Perché il parse degli argomenti di un tool in streaming va fatto solo alla fine del blocco? Quali errori sono ritentabili e quali no?

**Approfondimenti.** Esempi ufficiali Bedrock Runtime per Rust nella *AWS SDK for Rust Developer Guide*; API reference `Converse` e `ConverseStream`.

---

## Modulo 5 · Attivare Langfuse e vedere le chiamate

⏱ 4–6 ore.

**Obiettivo.** Un progetto Langfuse attivo (Cloud regione UE, oppure self-hosted in locale), l'app Rust che esporta ogni chiamata come *generation* con token, costo, modello e latenza, e tu che sai leggere una trace.

**Perché conta.** Da qui in avanti ogni esperimento del tutorial si legge in Langfuse, non nei log. Vedere le chiamate mentre impari accorcia il ciclo di apprendimento più di qualsiasi lettura. Ed è lo strumento che vuoi padroneggiare.

### Passi

- [ ] 5.1 📚 🔭 **Il modello dati.** Leggi *Observability Data Model* nella documentazione Langfuse. Da fissare: **trace** (una richiesta end-to-end; ha `input`, `output`, `user_id`, `session_id`, `tags`, `metadata`, `release`, `version`, `environment`), **observation** (span, generation, event, e i tipi agentici agent/tool/chain/retriever/evaluator; annidabili), **generation** (una chiamata al modello: modello, parametri, usage, costo, prompt collegato), **score** (numerico, categorico o booleano; su trace, observation o sessione), **session** (più trace di una stessa conversazione), **dataset** e **dataset run** (li vedrai nel Modulo 11).
- [ ] 5.2 🔭 **Scegli dove gira Langfuse.** Due strade, entrambe valide per imparare:
  - **Langfuse Cloud, regione UE** (`https://cloud.langfuse.com`, dati in Irlanda `eu-west-1`, la stessa area del tuo Bedrock). Zero infrastruttura, piano gratuito sufficiente per il tutorial.
  - **Self-hosted in locale** con Docker Compose: componenti `web`, `worker`, PostgreSQL, ClickHouse, Redis/Valkey, S3/MinIO. Utile per capire l'architettura (ingestione asincrona via coda, analitiche su ClickHouse) e per il Modulo 15, dove valuterai il self-hosting su AWS.
  Consiglio: parti dal Cloud UE, e fai il self-host locale come esperimento del Modulo 15. Scrivi la scelta in un ADR con la motivazione "residenza dei dati".
- [ ] 5.3 🔭 Crea organizzazione e **progetto** (`agentic-playground`). Genera una coppia di **API key** (public + secret) e mettile in `.env`. In Langfuse le chiavi sono per progetto: ambienti diversi (local, dev, prod) possono essere progetti diversi **oppure** lo stesso progetto con l'attributo `environment` sulle trace. Per il tutorial usa un progetto e `environment`.
- [ ] 5.4 🔭 **Modelli e prezzi.** Langfuse calcola il costo da `usage` solo se conosce il modello. In *Settings → Models* aggiungi le definizioni per i tuoi model id Bedrock (match per regex sul nome, per esempio `(?i)^eu\.anthropic\.claude-haiku-4-5.*`) con prezzi per token di input, output, cache read e cache write presi da 3.11. Senza questo passo vedrai i token ma non il costo.
- [ ] 5.5 💡 **Come si arriva a Langfuse da Rust.** Langfuse non ha un SDK ufficiale Rust; ha due porte aperte a qualunque linguaggio: l'**endpoint OpenTelemetry** (`/api/public/otel`, protocollo OTLP su HTTP, autenticazione Basic con `public:secret`) e l'**API pubblica REST**. Per il tracing userai OTel; per prompt, dataset, score ed esperimenti userai l'API. Crate utili: `opentelemetry-langfuse` (un builder per l'exporter OTLP già configurato per Langfuse), `langfuse-ergonomic` (client dell'API pubblica con builder; sopra a `langfuse-client-base`, generato dall'OpenAPI).
- [ ] 5.6 Nella crate `observability` inizializza lo stack: `tracing` → `tracing-opentelemetry` → `opentelemetry_sdk` con exporter OTLP verso Langfuse (`opentelemetry-otlp` con `endpoint = {LANGFUSE_HOST}/api/public/otel` e header `Authorization: Basic base64(public:secret)`, oppure il builder di `opentelemetry-langfuse`). Tieni anche il layer `fmt` su stdout per lo sviluppo. Batch exporter, flush esplicito allo shutdown (altrimenti perdi le ultime trace nei binari CLI).
- [ ] 5.7 🔭 **Mappare gli span sul modello dati.** Langfuse legge sia le convenzioni GenAI di OpenTelemetry sia i propri attributi `langfuse.*`. Regole minime:
  - Lo span **radice** diventa la trace: mettici `langfuse.session.id`, `langfuse.user.id` (già pseudonimizzato), `langfuse.trace.tags`, `langfuse.environment`, `langfuse.release` (versione dell'app), `langfuse.trace.input` e `langfuse.trace.output`.
  - Lo span di una chiamata LLM diventa una generation con `langfuse.observation.type = "generation"`, `gen_ai.request.model` (o `langfuse.observation.model.name`), `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, e i dettagli di cache in `langfuse.observation.usage_details` (JSON con `input`, `output`, `cache_read_input_tokens`, `cache_creation_input_tokens`); `langfuse.observation.input` e `.output` per i contenuti (vedi 5.10 per la privacy).
  - Gli span dei tool: `langfuse.observation.type = "tool"`; gli step della pipeline: `"span"` o `"chain"`.
  📚 Verifica i nomi esatti nella pagina *OpenTelemetry* della documentazione Langfuse: la lista degli attributi si evolve.
- [ ] 5.8 Scrivi un helper in `observability` che l'harness usa senza conoscere Langfuse: `record_generation(span, &LlmRequest, &LlmResponse)` che imposta gli attributi di 5.7 dai tuoi tipi neutri. Collegalo allo span di 4.10.
- [ ] 5.9 🧪 Esegui `app hello` tre volte. In Langfuse apri *Tracing → Traces*: devi vedere tre trace, ognuna con una generation, con modello, token, **costo** e latenza. Apri una generation: leggi input, output, usage. Se il costo manca, torna a 5.4. Se la trace manca, controlla flush e credenziali.
- [ ] 5.10 💡 🔭 **Contenuti e privacy, prima regola.** Decidi **ora** una politica per `input`/`output`: in `local` e `dev` completi; in `prod` passano da una funzione di redazione (Modulo 13) e da un campionamento. Implementala come configurazione di `observability`, non come `if` sparsi. Langfuse offre anche il mascheramento lato ingestione e la retention per progetto: annota entrambe le opzioni per il Modulo 13.
- [ ] 5.11 🧪 Simula una conversazione da CLI (`app chat` con tre messaggi): ogni messaggio è una trace, tutte con lo stesso `session.id`. In Langfuse apri *Sessions*: devi vedere la conversazione intera in ordine. Poi *Users*: devi vedere l'utente sintetico con costo cumulato.
- [ ] 5.12 🧪 Aggiungi il tuo primo **score** via API (`langfuse-ergonomic` o `POST /api/public/scores`): dopo `app hello`, attacca alla trace uno score booleano `smoke_ok = true`. Serve a capire il meccanismo che userai per guardrail (Modulo 9), feedback utente ed eval (Moduli 11 e 14).
- [ ] 5.13 🔭 Fai un giro completo dell'interfaccia e annota nel Diario a cosa serve ogni sezione: Tracing (Traces, Sessions, Users, Observations), Prompts, Evaluation (Datasets, Evaluators, Scores, Annotation Queues), Dashboards, Settings (Models, API keys, Members, Retention). È la mappa dei prossimi moduli.

**Fatto quando** ogni chiamata dell'app appare in Langfuse come generation con costo; le conversazioni sono raggruppate in sessioni; uno score arriva via API; la politica sui contenuti è in configurazione.

**Autoverifica.** Perché gli attributi di trace (`session.id`, `user.id`, `tags`) vanno messi sullo span radice e non solo su una generation? Che differenza c'è tra un progetto Langfuse per ambiente e un solo progetto con `environment`?

**Approfondimenti.** Pagina *OpenTelemetry* e *Observability Data Model* di Langfuse; README delle crate `opentelemetry-langfuse` e `langfuse-ergonomic`; *Self-hosting* per l'architettura v3.

---

## Modulo 6 · Prompt engineering sistematico con Langfuse Prompt Management

⏱ 8–10 ore, distribuite su più giorni.

**Obiettivo.** Un metodo ripetibile per scrivere, misurare e versionare i prompt, con Langfuse come registro dei prompt e i file in repo come fallback, applicato al prompt `SAFETY` del progetto guida.

**Perché conta.** "Il miglior prompt possibile" non esiste in astratto: esiste il miglior compromesso tra qualità, costo e latenza **per un task, un modello e un dataset misurato**. Senza misura si gira in tondo. E senza un registro con versioni non sai mai quale prompt ha prodotto quale risposta.

### Passi

- [ ] 6.1 📚 Leggi la guida di prompt engineering della documentazione Claude (sezione *Prompt engineering* su platform.claude.com) e le note di prompting per i modelli Claude 5: i prompt scritti per modelli vecchi tendono a essere troppo prescrittivi e peggiorano l'output dei modelli recenti.
- [ ] 6.2 💡 **Anatomia di un system prompt.** In ordine: (1) ruolo e contesto operativo, (2) obiettivo del task, (3) regole e vincoli (positivi: "fai X" funziona meglio di "non fare Y"), (4) formato di output, (5) esempi (few-shot) solo se servono, (6) dati variabili **alla fine**. I tag XML (`<contesto>`, `<regole>`, `<esempi>`) aiutano il modello a separare le parti e sono il modo standard per delimitare l'input dell'utente dal resto.
- [ ] 6.3 📚 🔭 **Prompt Management in Langfuse.** Leggi *Prompt Management Concepts*. Da fissare: un prompt ha un **nome**, **versioni** immutabili (1, 2, 3…), **label** che puntano a una versione (`production` è quella servita di default; `staging`, `latest` e label custom), tipo **text** o **chat**, **variabili** con sintassi `{{nome}}`, `config` (un JSON libero: ci metti modello, effort, `max_tokens`), tag, e la possibilità di comporre prompt dentro prompt. Le label `production` possono essere **protette** (solo certi ruoli le spostano).
- [ ] 6.4 🔭 Crea in Langfuse il prompt `safety-classifier` (tipo chat: un messaggio system con le regole e uno user con `{{message}}`), con `config = { "model_role": "fast", "effort": "low", "max_tokens": 200 }`. Assegna la label `production` alla v1.
- [ ] 6.5 🔭 Nella crate `prompts` implementa `PromptStore::get(name, label)`:
  1. chiama `GET /api/public/v2/prompts/{name}?label=production` (via `langfuse-ergonomic`),
  2. **cache in memoria** con TTL (60 secondi è il default degli SDK ufficiali; in produzione 5–10 minuti vanno bene),
  3. **fallback** al file `prompts/<name>.md` in repo se Langfuse non risponde (è il pattern "guaranteed availability" della documentazione),
  4. restituisce un oggetto con `name`, `version`, `template`, `config` e un metodo `render(vars)`.
  Un job `just prompts-pull` scarica in `prompts/` gli snapshot delle versioni `production`: così ogni cambiamento di prompt compare anche nel diff di una PR, e il fallback è sempre aggiornato.
- [ ] 6.6 🔭 **Collega prompt e generation.** Quando l'harness fa una chiamata usando un prompt, imposta sullo span della generation `langfuse.observation.prompt.name` e `langfuse.observation.prompt.version`. In Langfuse, aprendo il prompt, vedrai tutte le generation che lo hanno usato, con costo e latenza medie **per versione**: è la base di ogni confronto.
- [ ] 6.7 💡 **Stabilità del prefisso.** Ciò che non cambia tra chiamate (istruzioni, schema, esempi) va prima; ciò che cambia (profilo utente, contesto recuperato, messaggio) va dopo il `cachePoint`. Ogni byte che cambia nel prefisso azzera la cache. Con le variabili Langfuse questo si traduce in: variabili solo nella parte finale del template.
- [ ] 6.8 💡 **Structured output.** Definisci lo schema di output come struct Rust con `serde` + `schemars`, genera il JSON Schema e passalo in `outputConfig.textFormat` (Converse, tipo `json_schema`). In alternativa, forza un tool con quello schema. ⚠️ Non tutti i modelli supportano entrambe le strade, e Claude Fable 5.1 non accetta la forzatura del tool: fai un test per modello e registra il risultato nella `ModelRegistry` (Modulo 10). Lo schema vive nel codice, non nel prompt: nel prompt Langfuse metti solo il riferimento al nome dello schema.
- [ ] 6.9 💡 **Thinking ed effort.** Sui modelli Claude recenti il ragionamento è adattivo; regoli la profondità con `effort` (`low`, `medium`, `high`, `xhigh`, `max`). Per una classificazione (`SAFETY`) `low` è quasi sempre sufficiente; per la generazione principale prova `medium` e `high` e misura. 💡 Prima di passare a un modello più grande, prova il modello attuale con effort più alto: spesso costa meno. Il valore va nel `config` del prompt, così è versionato insieme al testo.
- [ ] 6.10 🧪 **Laboratorio: il prompt `SAFETY` in italiano.** Costruisci `datasets/safety/v1.jsonl` con **30–50 messaggi sintetici** etichettati a mano (`SAFE`, `PSYCH_CRISIS`, `MEDICAL_EMERGENCY`), includendo casi ambigui tipici dell'italiano ("non ce la faccio più", "sono stanca di tutto", "mi scoppia la testa"). Misura accuratezza, falsi negativi (il caso peggiore qui), latenza e costo su `BEDROCK_MODEL_FAST`. Per ora il runner è uno script semplice; nel Modulo 11 diventa un esperimento Langfuse.
- [ ] 6.11 🧪 🔭 Itera **in Langfuse**: v2 con esempi, v3 con regole più esplicite sul contesto, v4 con `effort` diverso nel `config`. Una variabile alla volta. Usa il **Playground** di Langfuse per provare rapidamente una variante su 3–4 casi prima di lanciare il dataset intero (collega il playground a Bedrock nelle impostazioni di *LLM connections*, se disponibile per il tuo modello; altrimenti prova dal tuo runner). Registra ogni run con la scheda dell'Appendice D. Sposta la label `production` solo sulla versione vincente.
- [ ] 6.12 🧪 Ripeti 6.10 con `BEDROCK_MODEL_MAIN`. Il modello grande batte il piccolo? Di quanto, a quale costo e latenza? Questa è la prima decisione da ADR: "per `SAFETY` usiamo X perché…".
- [ ] 6.13 💡 **Anti-pattern da riconoscere.** Prompt lunghissimi che "coprono tutti i casi"; regole in maiuscolo e minacce; chiedere JSON senza schema; mettere i dati dell'utente in mezzo alle istruzioni; cambiare prompt e modello insieme; giudicare un prompt su tre esempi; spostare `production` senza un dataset che lo giustifichi.
- [ ] 6.14 💡 **Prompt injection, prima nozione.** Tutto ciò che viene dall'utente o da un documento recuperato è **dato**, non istruzione. Delimitalo, e nel system prompt dichiara che il contenuto dentro certi tag va trattato come testo da analizzare. Approfondimento nel Modulo 13.

**Fatto quando** il prompt `SAFETY` vive in Langfuse con almeno tre versioni misurate sullo stesso dataset, l'app lo carica con cache e fallback, ogni generation in Langfuse mostra nome e versione del prompt usato, e un ADR dice quale versione e quale modello usi, con i numeri.

**Autoverifica.** Cosa succede all'app se Langfuse è irraggiungibile per un'ora? (Deve continuare a funzionare con i prompt in cache o dai file: se non è così, il fallback non è implementato.) Perché un few-shot di cinque esempi può peggiorare un classificatore?

---

## Modulo 7 · Tool calling e primo agent loop

⏱ 6–8 ore.

**Obiettivo.** Un harness generico in `harness`: registro di tool, loop di esecuzione, limiti di sicurezza, test con `FakeLlmClient`, trace annidate in Langfuse. Più un collegamento a un server MCP.

**Perché conta.** Questo è il pezzo che di solito un framework nasconde. Scriverlo a mano una volta ti fa capire cosa ti regala (e cosa ti toglie) un framework.

### Passi

- [ ] 7.1 💡 Definisci un `Tool` come trait: `name()`, `description()`, `input_schema()` (generato da `schemars` a partire da una struct di input), `async fn call(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>`. Un `ToolRegistry` li tiene per nome.
- [ ] 7.2 Traduci i `ToolSpec` in `ToolConfiguration` di Converse (nome, descrizione, `inputSchema.json`). In streaming e non.
- [ ] 7.3 💡 **Il loop.** Pseudocodice:

  ```text
  messages = [user]
  loop (max N iterazioni, budget di token e di tempo):
      resp = llm.complete(system, messages, tools)
      messages.push(resp.message)
      if resp.stop_reason != ToolUse: break
      results = esegui in parallelo tutti i ToolUse presenti nel messaggio
      messages.push(user message con TUTTI i ToolResult, nello stesso ordine)
  ```

  ⚠️ Tutti i `tool_result` di un turno vanno in **un solo** messaggio user; dividerli insegna al modello a non fare chiamate parallele. Un tool che fallisce restituisce un `tool_result` con `is_error = true`, mai un'eccezione che rompe il loop.
- [ ] 7.4 **Limiti.** Iterazioni massime, timeout complessivo, budget di token, lista di tool "pericolosi" che richiedono conferma umana (human-in-the-loop): il loop si sospende e restituisce uno stato `AwaitingApproval` che il chiamante può riprendere. Nel progetto guida non c'è ancora un tool pericoloso, ma il meccanismo serve.
- [ ] 7.5 📚 **Disegnare buoni tool.** Leggi *Writing effective tools for agents*. Regole pratiche: pochi tool con confini netti; nomi e descrizioni scritti per il modello (dicono *quando* usarli e *quando no*); input piccoli e tipizzati; output compatti (il risultato finisce nel contesto e si paga a ogni turno successivo); errori descrittivi che dicono al modello come correggersi.
- [ ] 7.6 🔭 **Trace annidate.** Lo span dell'agente (`langfuse.observation.type = "agent"`) contiene una generation per iterazione e uno span `"tool"` per ogni esecuzione di tool, con input e output. In Langfuse la trace deve leggersi come un albero: iterazione 1 → tool A, tool B → iterazione 2 → risposta finale. Aggiungi ai metadata della trace `iterations` e `tool_calls`.
- [ ] 7.7 🧪 Tool di esempio per il progetto guida: `search_knowledge_base(query, top_k)` (per ora su una lista in memoria con ricerca testuale; diventerà vettoriale nel Modulo 8), `get_user_profile(user_id)`, `get_current_date()`. Fai una demo in cui il modello decide da solo quando cercare, e leggila in Langfuse.
- [ ] 7.8 **Test senza rete.** Con `FakeLlmClient` scrivi test che verificano: il loop si ferma a `EndTurn`; un tool inesistente produce `is_error`; il tetto di iterazioni scatta; due `tool_use` nello stesso messaggio vengono eseguiti in parallelo e restituiti in un solo messaggio.
- [ ] 7.9 💡 **Model Context Protocol (MCP).** È lo standard per esporre tool a un modello attraverso un server separato. Con la crate ufficiale `rmcp` scrivi un **client** che si collega a un server MCP (per esempio un server filesystem di esempio, o uno tuo che espone la knowledge base) e adatta i suoi tool al tuo `Tool` trait. Così l'harness non distingue tool locali e remoti.
- [ ] 7.10 🧪 Misura in Langfuse quanti token costa il *solo* elenco dei tool nel prompt (confronta una generation con e senza tool). Se hai molti tool, questo costo si paga a ogni turno: è il motivo per cui esistono tecniche di caricamento su richiesta (Modulo 16).
- [ ] 7.11 💡 **Structured output via tool.** Implementa l'alternativa a 6.8: un tool `emit_result` con lo schema voluto. Confronta affidabilità e latenza con `outputConfig.textFormat`. Registra il risultato per modello.

**Fatto quando** il loop generico è testato senza rete, un tool MCP remoto è utilizzabile come uno locale, la trace in Langfuse mostra l'albero agente → generation → tool, e hai una demo in cui il modello cerca nella KB quando serve.

**Autoverifica.** Cosa succede se un tool restituisce 50 KB di testo? (Il contesto esplode, il costo cresce a ogni turno, la qualità cala.) Quali sono tre modi per evitarlo?

---

## Modulo 8 · Gestione del contesto (context engineering)

⏱ 8–10 ore.

**Obiettivo.** Decidere in modo esplicito **cosa** entra nel contesto di ogni chiamata, **in che ordine** e **con quale budget**: memoria breve, memoria lunga, retrieval, cache. E renderlo visibile in Langfuse.

**Perché conta.** Il contesto è una risorsa finita con rendimenti decrescenti. La qualità di un'applicazione agentica dipende più da cosa tieni fuori che da cosa metti dentro.

### Passi

- [ ] 8.1 📚 Leggi *Effective context engineering for AI agents*. Concetti da annotare: context rot, "la più piccola quantità di token ad alto segnale", just-in-time retrieval, compattazione, note strutturate, sub-agenti come strumento di isolamento del contesto.
- [ ] 8.2 💡 **Contare i token.** Bedrock espone l'operazione `CountTokens` (verifica la disponibilità per il tuo modello nella tua regione); in alternativa usa `usage.inputTokens` delle chiamate reali per calibrare una stima locale (caratteri / 3,5 per l'italiano). Registra sempre la stima **e** il valore reale: la differenza ti dice quanto puoi fidarti dell'euristica.
- [ ] 8.3 💡 **Budget esplicito.** Definisci per la chiamata `GENERATION` un budget, per esempio: system stabile ≤ 2.500 token, profilo compatto ≤ 400, contesto KB ≤ 1.500, turni recenti ≤ 3.000, messaggio corrente. Mettilo in configurazione, non nel codice. 🔭 Scrivi la ripartizione effettiva per sezione nei `metadata` della generation: in Langfuse potrai vedere dove finiscono i token quando il costo sale.
- [ ] 8.4 🧪 **Memoria breve: finestra a budget di token.** Implementa `conversation_log` append-only in SQLite e `recent_turns` **derivato** dal log: leggi i turni all'indietro accumulando i token finché il budget regge. Misura in token, non in numero di turni (chi scrive messaggi corti non va penalizzato). Test unitari con turni finti.
- [ ] 8.5 🧪 **Compattazione.** Quando la conversazione supera il budget, riassumi i turni più vecchi con una chiamata al modello economico e sostituiscili con un blocco `<riassunto_precedente>`. Definisci cosa il riassunto deve preservare (decisioni prese, argomenti aperti, tono) e verifica con un test che le informazioni chiave sopravvivano. 🔭 La compattazione è una generation a sé, con il suo prompt versionato in Langfuse (`conversation-summarizer`).
- [ ] 8.6 🧪 **Memoria lunga: il profilo utente.** Struct `UserProfile` con dimensioni, `confidence` e `last_updated`. La vista **compatta** iniettata in `GENERATION` filtra per confidenza e recenza (regola d'esempio dal documento ispiratore: confidenza > 0,5 e aggiornato negli ultimi 90 giorni, oppure confidenza > 0,8 sempre). L'aggiornamento (`PROFILE_UPDATE`) è asincrono, con structured output, e **atomico**: o sostituisce tutto il profilo o non tocca nulla.
- [ ] 8.7 💡 **Retrieval (RAG) essenziale.** Tre pezzi: embedding (su Bedrock: Amazon Titan Embeddings o Cohere Embed, tramite `invoke_model`), un vector store (in locale `pgvector` su PostgreSQL, oppure LanceDB/Qdrant), una funzione di ricerca con `top_k` e filtri per metadati. Il chunking dei documenti conta più del modello di embedding: chunk piccoli e con metadati (fonte, data di revisione, autore). 🔭 Il retrieval è uno span di tipo `retriever` con `query`, `top_k`, id e punteggi dei record trovati.
- [ ] 8.8 🧪 Sostituisci la KB in memoria del Modulo 7 con quella vettoriale. Usa 20–30 record sintetici. Misura *recall@3* su 15 query scritte a mano. Poi prova la variante gestita **Bedrock Knowledge Bases** per capire cosa ti toglie di lavoro e cosa ti toglie di controllo.
- [ ] 8.9 💡 **Ordine e cache.** Ricomponi il prompt di `GENERATION` così: [system stabile + schema + regole] → `cachePoint` → [profilo compatto] → [contesto KB] → [turni recenti] → [messaggio]. Verifica `cache_read_tokens` nelle chiamate consecutive dello stesso utente: in Langfuse compare nei dettagli di usage della generation e, se hai configurato il prezzo di cache read in 5.4, nel costo.
- [ ] 8.10 💡 **Pulizia del contesto agentico.** Nei loop con molti tool, i risultati vecchi diventano zavorra: sostituisci i `tool_result` più vecchi con un placeholder ("risultato rimosso, 1.240 token") dopo K turni. Misura l'effetto su costo e qualità.
- [ ] 8.11 Scrivi `docs/adr/000x-context-budget.md`: budget per sezione, regole della vista compatta, strategia di compattazione, motivazioni.

**Fatto quando** la finestra a budget, la compattazione e il profilo compatto sono testati senza rete, la KB vettoriale funziona, la cache viene letta nelle chiamate consecutive e i metadata delle generation in Langfuse mostrano la ripartizione dei token.

**Autoverifica.** Perché `recent_turns` deve essere derivato dal log e non scritto a parte? (Una sola fonte di verità, nessuna incoerenza da doppia scrittura.)

---

## Modulo 9 · La pipeline del progetto guida

⏱ 10–14 ore.

**Obiettivo.** Comporre i pezzi in una pipeline end-to-end funzionante da CLI e da HTTP, con una trace Langfuse per messaggio, e con streaming e guardrail in parallelo come **estensione** separata.

**Perché conta.** È il momento in cui i concetti diventano un sistema: concorrenza, cancellazione, retry, atomicità, e le domande di design che nessun tutorial sui prompt ti fa.

### Passi

- [ ] 9.1 💡 Modella la pipeline come **workflow esplicito**, non come agente libero: un `enum Stage` e una funzione per step, ognuna con input e output tipizzati. Il codice decide il flusso; l'LLM lavora dentro gli step. È il pattern giusto quando il flusso è noto in anticipo, e rende ogni step testabile da solo.
- [ ] 9.2 Implementa `STATE_READ` → `SAFETY` (con `RETRIEVAL` in parallelo via `tokio::join!`, il suo risultato si scarta se `SAFETY` blocca) → `GENERATION` → `GUARDRAILS` → `STATE_WRITE` → risposta → `PROFILE_UPDATE` in background (`tokio::spawn`, con tracing collegato allo span della richiesta).
- [ ] 9.3 🔭 **Una trace per messaggio.** Span radice = trace con `session.id` = conversazione, `user.id` pseudonimizzato, `tags = ["pipeline", ambiente]`, `input` = messaggio utente, `output` = risposta finale. Uno span figlio per step, con il nome dello step; le chiamate LLM sono generation dentro lo step. `PROFILE_UPDATE` gira dopo la risposta: fallo come span figlio ritardato **oppure** come trace separata con `metadata.parent_trace_id`; scegli e documenta. Ogni prompt usato porta nome e versione (6.6).
- [ ] 9.4 💡 **Escalation.** Le risposte di escalation di `SAFETY` sono **testi fissi** decisi da persone, non generati: nel dominio sanitario è una scelta di sicurezza e di responsabilità. Mettili in configurazione. 🔭 Tagga la trace `escalation:<tipo>`.
- [ ] 9.5 💡 🔭 **Loop evaluator-optimizer.** Se `GUARDRAILS` fallisce, rilancia `GENERATION` con il motivo del fallimento nel prompt; al massimo 2 tentativi, poi risposta di fallback sicura. Registra l'esito come **score** sulla trace (`guardrail_pass` booleano, `guardrail_failure_type` categorico) e il numero di tentativi nei metadata: la frequenza dei fallimenti diventa un grafico e un alert nel Modulo 14.
- [ ] 9.6 💡 **Robustezza.** Timeout per step; retry con backoff e jitter solo su errori ritentabili; idempotenza degli scritti (una richiesta ripetuta non duplica turni); `PROFILE_UPDATE` atomico con rollback; cancellazione pulita se il client chiude la connessione (`CancellationToken`). Gli errori diventano span con `level = ERROR` e `status_message`, così in Langfuse li filtri.
- [ ] 9.7 Esponi la pipeline via HTTP con `axum`: `POST /chat` (risposta completa) e `POST /chat/stream` (SSE). Autenticazione finta per ora (header con `user_id`). Restituisci nel corpo della risposta il `trace_id`: servirà per collegare il feedback dell'utente (Modulo 14).
- [ ] 9.8 🧪 Misura in Langfuse la latenza per step su 20 messaggi sintetici: `SAFETY`, `GENERATION` (time-to-first-token e totale), `GUARDRAILS`. Confronta con le stime del documento ispiratore. Annota costo per messaggio (Langfuse lo somma per trace).
- [ ] 9.9 🧪 **Estensione: streaming con guardrail in parallelo.** Inoltra i token di `GENERATION` al client man mano, mentre `GUARDRAILS` analizza il testo accumulato (a chunk, o a fine stream). Se fallisce, invii un evento `retract` e la risposta corretta. ⚠️ È un epic a sé: buffering, abort pulito, UX del "testo che sparisce". Fallo solo dopo che la versione sincrona è stabile e misurata, e documenta il compromesso in un ADR.
- [ ] 9.10 💡 **Alternativa agentica.** Riscrivi (in un branch) `GENERATION` come agente con tool (`search_knowledge_base`, `get_user_profile`) invece che con contesto pre-iniettato. Confronta in Langfuse qualità, costo, latenza e prevedibilità. Questo confronto è la lezione centrale di "workflow vs agente".
- [ ] 9.11 ADR: struttura della pipeline, scelte di concorrenza, politica di retry, scelta workflow vs agente, forma della trace.

**Fatto quando** `POST /chat` risponde end-to-end con dati sintetici, ogni messaggio produce in Langfuse una trace con uno span per step e gli score dei guardrail, il profilo si aggiorna in background senza corrompersi in caso di errore, e hai una tabella di latenza e costo per step.

**Autoverifica.** Perché `RETRIEVAL` può partire in parallelo a `SAFETY` ma `GENERATION` no? Cosa rende un aggiornamento "atomico" nel tuo codice?

---

## Modulo 10 · Router di modelli

⏱ 8–10 ore.

**Obiettivo.** Una crate `router` che sceglie il modello per ogni chiamata secondo regole esplicite, gestisce fallback e degradazione, ed espone le decisioni in Langfuse.

**Perché conta.** Ogni step ha esigenze diverse (una classificazione non ha bisogno del modello più grande). Il router è dove costo, latenza, qualità e disponibilità si incontrano. Fatto male, nasconde i problemi; fatto bene, è la leva di ottimizzazione più economica.

### Passi

- [ ] 10.1 💡 **ModelRegistry.** Una tabella di `ModelSpec` caricata da configurazione (TOML): `id`, `provider`, `tier` (small/mid/large), prezzo input/output/cache per Mtok, context window, `supports_tools`, `supports_structured_output`, `supports_streaming`, `region_profile`, quota (rpm, tpm), latenza p50 misurata. Le capacità le **verifichi con test** (Modulo 12), non le copi dalla documentazione. I prezzi devono coincidere con quelli inseriti in Langfuse (5.4): tienili in un solo file e generane entrambi.
- [ ] 10.2 💡 **Strategie di routing**, da implementare in ordine crescente di complessità:
  1. **Statica per step**: `SAFETY → small`, `GENERATION → mid/large`, `GUARDRAILS → small`, `PROFILE_UPDATE → small`. Copre il 90% dei casi reali. Il `config` del prompt in Langfuse (6.4) dice il **ruolo** (`fast`, `main`), il router traduce il ruolo in modello: così cambiare modello non richiede toccare il prompt e viceversa.
  2. **A regole**: per lunghezza input, lingua, presenza di tool, richiesta di structured output, utente premium.
  3. **A cascata**: prova il modello economico; se la confidenza dichiarata è bassa o l'output non valida lo schema, passa al modello superiore. Misura quanto spesso scala.
  4. **Con classificatore**: un piccolo modello (o una regola) stima la difficoltà e sceglie il tier. Attenzione al costo della chiamata in più.
  5. **Gestita**: *Amazon Bedrock Intelligent Prompt Routing* sceglie tra modelli di una famiglia in base al prompt. Provala per capire cosa fa, e confrontala con la tua strategia 3.
- [ ] 10.3 💡 **Fallback e resilienza.** Per ogni ruolo una catena: primario → secondario (stesso tier, altro modello o altra regione EU) → degradazione (risposta di cortesia, coda). Retry solo su errori ritentabili, con backoff e jitter (l'SDK lo fa per la singola chiamata; il router lo fa **tra modelli**). Circuit breaker per modello: dopo N errori in T secondi, salta al secondario per un periodo, poi riprova gradualmente.
- [ ] 10.4 💡 **Vincoli da rispettare.** Timeout residuo della richiesta (non ha senso fare fallback se restano 200 ms); quote per modello (un token bucket locale con `governor` evita di andare in throttling); regione (mai uscire da EU); capacità (non instradare una richiesta con tool a un modello che non li supporta).
- [ ] 10.5 ⚠️ **La cache è per modello.** Cambiare modello a metà conversazione azzera il prompt cache e può cambiare stile e comportamento. Il router deve preferire la **stabilità per sessione**: fissa il modello all'inizio della conversazione e cambialo solo per errori, non per ottimizzazione istantanea.
- [ ] 10.6 🔭 **Il router in Langfuse.** Ogni generation porta nei metadata `router.role`, `router.reason` (`static`, `fallback:throttled`, `escalation:low_confidence`), `router.attempt`; il modello effettivo è già nel campo modello. In Langfuse crea un **dashboard** con: distribuzione delle generation per modello, costo per ruolo, conteggio dei fallback (filtro sui metadata). Vedrai subito se un modello economico "scala" troppo spesso.
- [ ] 10.7 💡 **Sperimentazione controllata.** Supporta un `override` per richiesta (header o config) per forzare un modello: serve per gli esperimenti del Modulo 11. Aggiungi *shadow routing*: la richiesta va al modello A, e un campione (per esempio il 5%) va **anche** a B in background solo per confronto, senza toccare la risposta all'utente; la generation "ombra" ha il tag `shadow`.
- [ ] 10.8 🧪 Simula guasti con `FakeLlmClient` (throttling a raffica, timeout, output non valido) e verifica: fallback corretto, circuit breaker che apre e richiude, nessuna richiesta oltre la quota configurata.
- [ ] 10.9 🧪 Con modelli reali: confronta la strategia statica con la cascata su 50 messaggi sintetici. Costo totale, latenza p50/p95, qualità (Modulo 11). Scrivi l'ADR con la scelta.

**Fatto quando** il router è configurato da file, testato sui guasti senza rete, e ogni decisione è leggibile in Langfuse con la sua motivazione.

**Autoverifica.** Perché "il modello più economico che passa i test" non è la regola giusta per `GENERATION` se il costo per **task completato** (non per chiamata) è più alto per via di retry e rigenerazioni?

**Approfondimenti.** Pagina *Intelligent Prompt Routing* di Bedrock; il pattern "orchestrator-workers" per delegare sotto-task a modelli piccoli senza cambiare modello nel loop principale.

---

## Modulo 11 · Valutare i modelli con dataset ed esperimenti

⏱ 10–12 ore.

**Obiettivo.** Un runner in `evals` che esegue un **dataset Langfuse** con una configurazione (modello, prompt@versione, effort), registra i risultati come **dataset run** con score, e produce un report confrontabile: qualità con intervallo di confidenza, costo, latenza. Più i giudici LLM gestiti da Langfuse e la calibrazione con annotazioni umane.

**Perché conta.** Senza eval decidi a sensazione. Con un eval **rotto** decidi con sicurezza nella direzione sbagliata. La maggior parte dei "risultati sorprendenti" sono bug dell'eval.

### Passi

- [ ] 11.1 📚 🔭 Leggi *LLM Evaluation Concepts* e *Evaluation Overview* di Langfuse. Da fissare: **dataset** (collezione di item con `input`, `expected_output`, `metadata`), **dataset run** (un esperimento: ogni item eseguito produce una trace collegata all'item, più score), **evaluator** (un giudice LLM o una funzione che assegna score a trace di produzione o di esperimento), **annotation queue** (coda per far etichettare a una persona), **score** con tre tipi (numerico, categorico, booleano).
- [ ] 11.2 💡 **Cosa valuti, per step.** `SAFETY`: accuratezza, e soprattutto **falsi negativi** (un rischio non visto). `GENERATION`: aderenza allo schema, uso corretto dei riferimenti KB (niente affermazioni non presenti nei record citati), tono, qualità dell'italiano, assenza di diagnosi. `GUARDRAILS`: accordo con giudizio umano. `PROFILE_UPDATE`: correttezza dell'estrazione. Ogni step ha il suo dataset.
- [ ] 11.3 🔭 **Dataset.** Sorgente di verità: i file JSONL in `datasets/` (versionati in git). Un comando `just datasets-push` li carica in Langfuse come dataset (`safety-v1`, `generation-v1`…) via API. Da dove vengono i casi: scritti a mano, sintetizzati con un modello **e poi rivisti da una persona**, o (in produzione) da trace reali aggiunte al dataset con un clic dall'interfaccia Langfuse. ⚠️ Mai usare come "verità" l'output del modello che stai valutando. Bilancia le classi. Per `SAFETY` inserisci esplicitamente casi in entrambe le direzioni (deve scattare / non deve scattare).
- [ ] 11.4 💡 **Tipi di grader.** (1) Programmatico: exact match, validazione schema, regex, controllo che gli ID KB citati esistano. (2) Rubrica con LLM-judge: proprietà **atomiche** valutate una per una (`no_diagnosis`, `grounded_in_kb`, `tone_ok`), output strutturato, con testo candidato trattato come dato e non come istruzione. (3) Pairwise: due risposte, quale è migliore (randomizza l'ordine A/B). Usa un giudice di **famiglia diversa** dal modello valutato quando possibile.
- [ ] 11.5 🔭 **Il runner come dataset run.** `just eval --dataset safety-v1 --model X --prompt safety-classifier@v3 --reps 3` fa: crea un run con nome parlante (`safety-v1 / haiku / v3 / 2026-09-20`), per ogni item e ripetizione esegue la pipeline reale dello step (non una copia), collega la trace all'item del dataset, calcola i grader programmatici e li scrive come score sul run. Metadata del run: modello, prompt@versione, effort, commit git.
- [ ] 11.6 🔭 **Giudici gestiti da Langfuse.** In *Evaluation → Evaluators* crea un evaluator LLM-as-a-judge per `no_diagnosis` e uno per `grounded_in_kb`, con la tua rubrica, collegato a Bedrock, filtrato sulle trace degli esperimenti. Langfuse li esegue da solo su ogni nuova trace che matcha. Confronta con l'alternativa "giudice nel runner": il primo è comodo e riusabile in produzione (Modulo 14), il secondo è testabile e versionato in git. Puoi tenerli entrambi.
- [ ] 11.7 🔭 **Calibrare il giudice.** Manda 30–50 trace in un'**annotation queue**, etichettale tu (o un collega clinico) con gli stessi score del giudice, poi confronta accordo giudice/umano. Sotto il 90% su casi chiari, la rubrica va rifatta. Ripeti quando cambi modello giudice.
- [ ] 11.8 💡 **Metriche di performance dall'API, non stimate.** Token in/out/cache dalla risposta; costo calcolato dal listino del modello **che ha risposto davvero**; latenza della sola chiamata riuscita (senza i retry). Il costo del giudice è visibile a parte in Langfuse perché le sue generation sono trace distinte.
- [ ] 11.9 💡 **Ripetizioni e rumore.** Esegui ogni caso R volte (almeno 2–3). La metà larghezza dell'intervallo di confidenza su un tasso di successo è circa `1/sqrt(n·R)`: 30 casi × 2 ripetizioni ≈ ±13 punti. Se la differenza tra due run è sotto il rumore, **non c'è differenza**. Il runner stampa questo numero nel report; Langfuse mostra le medie per run, non gli intervalli: calcolali tu.
- [ ] 11.10 💡 **Igiene dell'harness di eval.** Distingui errori di infrastruttura (timeout, throttling, output troncato a `max_tokens`) da errori del modello: i primi diventano score `infra_error = true` e non entrano nella media di qualità. La traiettoria completa è già in Langfuse (trace collegata all'item). Verifica che il modello che ha risposto sia quello richiesto. Esegui un **oracolo** (le risposte attese devono passare) e un **baseline nullo** (risposta vuota deve fallire): se non succede, l'eval è rotto.
- [ ] 11.11 🧪 🔭 **Confronto modelli per `SAFETY`.** Tre run (Haiku, Sonnet, un non-Anthropic) sullo stesso dataset e prompt. In Langfuse usa la vista di **confronto tra run** del dataset: score medi, costo, latenza fianco a fianco. Poi apri i casi in cui i run discordano: sono i più istruttivi. Decisione nell'ADR.
- [ ] 11.12 🧪 **Confronto per `GENERATION`.** Grader misto: schema + grounding programmatici, rubrica LLM-judge per tono e assenza di diagnosi. Prova anche lo stesso modello con `effort` diverso: spesso è la variabile più economica.
- [ ] 11.13 🔭 **Esperimenti dall'interfaccia.** Langfuse permette di lanciare un esperimento su un dataset direttamente dall'UI (prompt × modello) senza codice. Provalo su `safety-v1`: è utile per iterare sui prompt senza toccare il runner, ma non esegue la **tua** pipeline (schema, router, tool). Annota quando usare l'uno e quando l'altro.
- [ ] 11.14 💡 Prova anche **Amazon Bedrock Evaluations** (giudice LLM gestito da AWS) per capire cosa offre il provider. Confronta con Langfuse: dove vivono i dati, quanto controllo hai sulla rubrica, come si integra con la pipeline reale.
- [ ] 11.15 💡 **L'eval è vivo.** Ogni bug trovato in produzione diventa un item (da Langfuse: trace → "aggiungi a dataset"). Ogni item su cui tutti i modelli fanno 100% va reso più difficile o archiviato. Ricalibra il giudice quando cambi modello giudice.

**Fatto quando** i dataset vivono in git e in Langfuse in modo sincronizzato, il runner produce dataset run con score e un report con intervalli di confidenza, un evaluator gestito gira sulle trace degli esperimenti ed è calibrato contro annotazioni umane, e hai due ADR con la scelta dei modelli per `SAFETY` e `GENERATION`.

**Autoverifica.** Se il run A fa 92% e il run B 88% su 25 casi con 1 ripetizione, cosa puoi concludere? (Niente: il rumore è ±20 punti.) Perché il runner deve chiamare la pipeline reale e non una copia semplificata?

---

## Modulo 12 · Suite di test anti-regressione

⏱ 8–10 ore.

**Obiettivo.** Una piramide di test che protegge l'harness (deterministico) e l'integrazione con i modelli (non deterministica) con strumenti diversi e cadenze diverse, con Langfuse come registro dei risultati degli esperimenti di regressione.

**Perché conta.** I modelli cambiano sotto i tuoi piedi (nuove versioni, cambi di comportamento del provider), i prompt cambiano per mano tua (e ora anche da un'interfaccia web), l'harness cambia per refactoring. Servono reti diverse.

### Passi

- [ ] 12.1 💡 **Livello 1: unit test dell'harness (PR, senza rete).** Tutto ciò che è logica pura: rendering dei prompt, parsing dell'output strutturato, finestra a budget di token, regole della vista compatta, decisioni del router, loop dei tool, fallback del `PromptStore`. Usa `FakeLlmClient` e un `FakePromptStore`. Devono essere centinaia e girare in secondi con `cargo nextest`.
- [ ] 12.2 💡 🔭 **Snapshot test dei prompt.** Con `insta`, uno snapshot del prompt **renderizzato** per ogni step con input fissi, a partire dagli snapshot in `prompts/` scaricati da Langfuse (6.5). Se qualcuno sposta la label `production` in Langfuse, il prossimo `just prompts-pull` cambia il file, lo snapshot fallisce e la modifica compare nel diff della PR: è il tuo controllo di revisione sui prompt modificati dall'interfaccia.
- [ ] 12.3 💡 **Livello 2: test dell'adapter Bedrock (PR, senza rete).** Con `aws-smithy-mocks` (o `StaticReplayClient` dell'SDK) simuli le risposte del servizio: verifichi la mappatura dei tipi, la gestione degli eventi di streaming, la classificazione degli errori (throttling, validazione, accesso negato), il comportamento dei retry. Registra risposte reali una volta e usale come fixture. Stesso approccio per il client Langfuse con `wiremock`.
- [ ] 12.4 💡 **Livello 3: contract test con i servizi veri (nightly, `#[ignore]`).** Pochi test, uno per capacità dichiarata nella `ModelRegistry`: il modello risponde; supporta i tool; supporta lo structured output; il cache read funziona; lo streaming emette gli eventi attesi; `max_tokens` piccolo produce `stop_reason = max_tokens`. Più due per Langfuse: una trace inviata via OTel compare via API entro N secondi; un prompt `production` si scarica. Girano con `cargo nextest run --run-ignored ignored-only` e credenziali fornite dalla CI via OIDC (mai access key statiche nei secret).
- [ ] 12.5 💡 🔭 **Livello 4: esperimenti di regressione (nightly o su cambi di prompt/modello).** La nightly lancia il runner del Modulo 11 sui dataset di regressione e poi legge gli score del run via API e li confronta con **soglie**: `safety.false_negative_rate ≤ 2%`, `generation.schema_valid ≥ 99%`, `generation.grounded ≥ 95%`, costo medio per messaggio ≤ X. La soglia va fissata **sopra il rumore** stimato: altrimenti il test è un generatore di falsi allarmi. Fallimento = build rossa, con link al run in Langfuse nel messaggio. Un **alert** Langfuse sullo score medio del run (Modulo 14) è la seconda rete.
- [ ] 12.6 💡 **Non determinismo.** Nei test di livello 3 e 4 non asserire mai su testo esatto: asserisci proprietà (schema valido, campo presente, valore in un insieme), su più ripetizioni, oltre una soglia. Segna i test soggetti a flakiness e traccia il loro tasso di fallimento: se sale, è un segnale, non un fastidio.
- [ ] 12.7 💡 **Pinning e canary.** Fissa le versioni dei modelli dove il provider lo consente; quando arriva una versione nuova, fai girare l'intera suite di livello 3 e 4 sulla nuova versione **prima** di cambiare la registry (è il tuo canary): in Langfuse il run nuovo si confronta con l'ultimo run buono. Documenta la migrazione in un ADR.
- [ ] 12.8 Property-based test con `proptest` per le funzioni "ostili": finestra a budget con turni di lunghezza casuale, parser dell'output strutturato con JSON quasi valido, tokenizzazione approssimata.
- [ ] 12.9 Test di sicurezza come regressione: un dataset `injection-v1` di tentativi di prompt injection che **non devono** cambiare il comportamento di `GENERATION` né bypassare `SAFETY` (Modulo 13). È un dataset Langfuse come gli altri.
- [ ] 12.10 Collega tutto alla CI: `ci.yml` esegue livelli 1–2; `nightly.yml` esegue 3–4 e pubblica il report con i link ai run; un badge nel README.

**Fatto quando** una modifica di prompt in Langfuse fa fallire uno snapshot in PR dopo il pull; un mock di throttling è gestito correttamente; la nightly gira sui servizi veri, produce dataset run in Langfuse e li confronta con soglie.

**Autoverifica.** Perché un test di livello 4 con soglia al 95% su 20 casi e 1 ripetizione è un test inutile? Cosa protegge dal rischio che qualcuno sposti `production` su un prompt non testato?

---

## Modulo 13 · Sicurezza, privacy e guardrail

⏱ 6–8 ore.

**Obiettivo.** Difese a strati contro input ostili, fuga di dati e comportamenti fuori scopo; conformità di base per dati sanitari in UE, inclusi i dati che finiscono in Langfuse.

**Perché conta.** Un'applicazione agentica ha una superficie d'attacco nuova: l'input dell'utente e i documenti recuperati sono testo che il modello potrebbe interpretare come istruzioni. Nel sanitario un errore è un danno reale. E lo strumento di osservabilità, per sua natura, **copia** i dati: va governato come il sistema che osserva.

### Passi

- [ ] 13.1 📚 Leggi la *OWASP Top 10 for LLM Applications*. Per ognuna delle dieci voci scrivi se e come tocca il progetto guida.
- [ ] 13.2 💡 **Prompt injection.** Difese: delimitazione chiara dei dati (tag XML), istruzioni nel system prompt sul trattare i contenuti come dati, privilegi minimi ai tool (un tool non può fare più di quanto serve), conferma umana per azioni irreversibili, validazione dell'output prima di eseguirlo o mostrarlo. Nessuna difesa è completa: per questo servono più strati e il dataset di regressione (12.9).
- [ ] 13.3 💡 **Bedrock Guardrails.** Crea un guardrail con: filtri di contenuto, argomenti negati (per esempio "diagnosi"), filtro PII con mascheramento, controllo di grounding (l'output è supportato dalla fonte?). Applicalo con `ApplyGuardrail` a input e output come strato **aggiuntivo** al tuo `GUARDRAILS` LLM. Confronta: cosa prende uno che l'altro non prende? Costo e latenza aggiunti? 🔭 L'esito è uno score in più sulla trace (`bedrock_guardrail_action`).
- [ ] 13.4 💡 🔭 **Dati personali nel prompt e nelle trace.** Minimizza ciò che entra nel prompt; pseudonimizza gli identificativi (lo `user.id` in Langfuse non deve mai essere un identificativo reale); nelle trace applica la **redazione della PII prima dell'esportazione** (la funzione di 5.10: nomi, telefoni, email, codici fiscali; per il testo libero valuta un modello piccolo o le API di Bedrock Guardrails per la PII); definisci la **retention** per progetto in Langfuse; documenta il flusso dei dati (chi vede cosa, in che regione). Bedrock non usa i tuoi dati per addestrare modelli e non li conserva oltre la richiesta, salvo logging attivato da te: rivedi la scelta di 3.9.
- [ ] 13.5 💡 🔭 **Residenza dei dati.** Solo inference profile `eu.*`; nessuna funzione che instrada fuori dall'UE; Langfuse Cloud regione UE oppure self-hosted in AWS UE (Modulo 15); verifica dove finiscono log, trace, dataset, export. Scrivi tutto in un ADR: sarà la base per la revisione legale.
- [ ] 13.6 💡 **IAM a privilegio minimo.** Sostituisci `AmazonBedrockFullAccess` con una policy sugli ARN degli inference profile usati; ruoli distinti per app, CI ed eval; nessuna access key statica in produzione (ruoli IAM per ECS/Lambda, OIDC per GitHub Actions).
- [ ] 13.7 💡 🔭 **Segreti e accessi Langfuse.** API key Langfuse in Secrets Manager, distinte per app, CI e runner; in Langfuse, membri con ruoli minimi (chi può spostare `production` sui prompt, chi può vedere le trace con contenuti); label `production` **protetta**.
- [ ] 13.8 💡 **Abuso e costi.** Rate limit per utente (`governor` in axum), lunghezza massima del messaggio, tetto di costo per sessione e per giorno, alert su anomalie (un utente che genera il 30% del costo: in Langfuse la vista *Users* lo mostra).
- [ ] 13.9 💡 **Output verso l'utente.** Le risposte di escalation sono fisse (9.4). Le risposte generate passano da `GUARDRAILS`, da Bedrock Guardrails e da una validazione programmatica (lunghezza, lingua, niente URL non in whitelist).
- [ ] 13.10 ADR: modello delle minacce, controlli per strato, cosa resta scoperto e perché.

**Fatto quando** il dataset di injection passa, Bedrock Guardrails è integrato e misurato, IAM è a privilegio minimo, le trace in `prod` sono redatte, la retention è impostata e l'ADR sui dati è scritto.

**Autoverifica.** Perché "scrivo nel prompt di ignorare le istruzioni contenute nei documenti" non basta come difesa? Perché il tuo strumento di osservabilità va trattato come un sistema che elabora dati personali?

---

## Modulo 14 · Osservabilità in produzione con Langfuse

⏱ 8–10 ore.

**Obiettivo.** Langfuse come centro operativo: ogni richiesta è una trace con uno span per step; dashboard e alert sulle metriche che contano; valutazione **online** con giudici gestiti su un campione del traffico; feedback degli utenti come score; code di annotazione per la revisione umana; correlazione con il resto dell'infrastruttura.

**Perché conta.** In produzione non puoi leggere ogni conversazione. Devi sapere: quanto costa, quanto è lento, quanto spesso i guardrail scattano, quando un modello cambia comportamento, e poter ricostruire una singola conversazione problematica in pochi minuti.

### Passi

- [ ] 14.1 💡 **Cosa osservare.** Per richiesta: `trace_id`, `session_id`, `user_id` pseudonimizzato, step, modello scelto e motivo, prompt name+version, token in/out/cache, costo, latenza (time-to-first-token e totale), `stop_reason`, esito guardrail, errori classificati, tool chiamati con durata. Aggregati: costo/giorno e per ruolo, p50/p95 latenza per step, tasso errori per modello, tasso fallimenti guardrail, tasso fallback del router, cache hit rate, distribuzione dei modelli, score di qualità online. Quasi tutto è già nelle trace dei Moduli 5–10: qui lo rendi leggibile.
- [ ] 14.2 🔭 **Ambienti e release.** Ogni trace porta `environment` (`prod`, `staging`) e `release` (versione o commit dell'app). Così ogni grafico si filtra per ambiente e ogni cambiamento di metrica si allinea a un deploy.
- [ ] 14.3 🔭 **Dashboard.** In *Dashboards* costruisci una dashboard "Pipeline" con: costo giornaliero per ruolo del router; p95 latenza per step (filtro sul nome dello span); conteggio trace per tag `escalation:*`; tasso `guardrail_pass` (la media di uno score booleano è la percentuale di veri); tasso fallback (metadata `router.reason` che inizia con `fallback`); cache read su input totale; distribuzione delle generation per modello. Aggiungi la vista **Pulse** sulla tabella delle observation per individuare outlier di costo e latenza.
- [ ] 14.4 🔭 **Alert.** In Langfuse crea alert a soglia con notifica Slack o webhook su: `guardrail_pass` medio sotto soglia nell'ultima ora (un salto improvviso spesso significa che il provider ha cambiato modello o che qualcuno ha spostato `production` su un prompt); costo giornaliero sopra budget; p95 di `GENERATION` sopra soglia; tasso di errori per modello. Collega gli alert degli evaluator (14.6) direttamente dalla pagina dell'evaluator.
- [ ] 14.5 🔭 **Costi riconciliati.** Il costo che Langfuse calcola (dai token e dai prezzi di 5.4) va confrontato mensilmente con la bolletta AWS filtrata per il tag di 3.8 e per **Application Inference Profile** (crea un profilo applicativo per ambiente: Cost Explorer mostra il costo per profilo). Se non coincidono entro il 5%, un prezzo o un model id nella registry è sbagliato.
- [ ] 14.6 🔭 **Valutazione online.** Gli evaluator LLM-as-a-judge del Modulo 11 girano anche in produzione: configurali con un filtro (`environment = prod`, campionamento per esempio del 5%, esclusi gli step non pertinenti) così Langfuse assegna `no_diagnosis` e `grounded_in_kb` a un campione del traffico reale. Il costo del giudice è misurabile e va tenuto sotto controllo con il campionamento. Aggiungi anche evaluator **a codice** (per esempio: la risposta cita solo ID KB esistenti) dove non serve un modello.
- [ ] 14.7 🔭 **Feedback degli utenti.** Il client invia pollice su/giù (e un commento opzionale) con il `trace_id` ricevuto in 9.7; l'app lo scrive come score `user_feedback` sulla trace via API. In Langfuse filtra le trace con feedback negativo: sono la prima fonte di nuovi item per i dataset (11.15).
- [ ] 14.8 🔭 **Revisione umana continua.** Un'annotation queue "revisione clinica settimanale" alimentata automaticamente: trace con `guardrail_pass = false`, con feedback negativo, o con score del giudice sotto soglia. Una persona le etichetta; il confronto tra etichette umane e giudice tiene il giudice calibrato nel tempo (11.7).
- [ ] 14.9 💡 **Drift.** Confronta settimanalmente gli score online e le distribuzioni (lunghezza risposte, tasso escalation, costo per messaggio) con la baseline della release precedente: un cambiamento senza deploy tuo è quasi sempre un cambiamento del modello o di un prompt. Il confronto tra dataset run (12.7) conferma o smentisce.
- [ ] 14.10 💡 **Il resto dell'infrastruttura.** Langfuse è specializzato sull'LLM; CPU, memoria, database, code e latenza HTTP vivono altrove (in azienda: Datadog). Due modi per correlare: (a) usare **OpenTelemetry Collector** come fan-out, l'app manda le trace al collector e il collector le invia sia a Langfuse sia a Datadog (le stesse trace, con `trace_id` identico, in due posti); (b) esportare da Langfuse le metriche aggregate via **Metrics API** verso il sistema di monitoraggio esistente. Prova (a): è il pattern che ti permette di cambiare backend senza toccare l'app.
- [ ] 14.11 💡 🔭 **Volume e costi di Langfuse.** Stima trace/mese e osservation/trace; verifica i limiti del piano Cloud o il dimensionamento del self-host; imposta la retention; usa l'**export batch** (verso S3) per l'analisi storica offline.
- [ ] 14.12 Scrivi un runbook breve: "la latenza sale" → cosa guardare in Langfuse e in Datadog; "i guardrail scattano di più" → confronta prompt version e modello delle ultime ore; "il costo raddoppia" → dashboard costo per ruolo, poi *Users*; "lo score online cala" → ultimo run di regressione, ultimo deploy, ultimo spostamento di label.
- [ ] 14.13 🧪 Prova a debuggare una conversazione: da un `trace_id` ricostruisci tutti gli step, i prompt esatti (con versione, cliccabili dalla generation), le decisioni del router e il costo. Se non ci riesci in cinque minuti, manca qualcosa nella strumentazione.

**Fatto quando** esiste una dashboard "Pipeline" con alert attivi, un evaluator online gira su un campione di `prod`, il feedback utente arriva come score, un'annotation queue si riempie da sola, e il runbook è scritto.

**Autoverifica.** Perché il tasso `guardrail_pass` è un ottimo "canarino" per i cambiamenti silenziosi del provider? Perché il campionamento è indispensabile per la valutazione online?

**Approfondimenti.** Documentazione Langfuse: *Evaluation* (evaluators su traffico live), *Dashboards*, *Alerts*, *Annotation Queues*, *Metrics API*, *Public API*.

---

## Modulo 15 · Deploy in produzione

⏱ 8–12 ore.

**Obiettivo.** L'applicazione gira su AWS con ruoli IAM (non chiavi), configurazione per ambiente, streaming funzionante, rollout graduale, controlli di costo; e una decisione motivata su dove gira Langfuse.

**Perché conta.** Molte scelte fatte finora (streaming, timeout, cache, quote, volume di trace) si misurano solo con un deploy reale e un po' di carico.

### Passi

- [ ] 15.1 **Packaging.** Dockerfile multi-stage per Rust (build in un'immagine con toolchain, runtime `distroless` o `debian-slim`), binario ottimizzato (`--release`, LTO), immagine piccola. Scansione vulnerabilità (`cargo deny`, scanner immagini).
- [ ] 15.2 💡 **Configurazione a 12 fattori.** Tutto da variabili d'ambiente o Parameter Store: regione, model id per ruolo, budget di contesto, soglie del router, host e chiavi Langfuse, politica di tracing (contenuti, campionamento). Nessun `if env == prod` nel codice.
- [ ] 15.3 💡 **Dove eseguire l'app.** Valuta tre opzioni e scegli con un ADR: **ECS Fargate** (servizio HTTP sempre acceso, streaming SSE naturale, il default sensato); **AWS Lambda** con `cargo-lambda` (ottimo per carichi a picchi; ⚠️ streaming delle risposte, timeout e **flush delle trace OTel** prima della fine dell'invocazione vanno verificati); **Bedrock AgentCore Runtime** (ambiente gestito per agenti, con Gateway per i tool, Memory, Policy con Guardrails, Observability integrata: utile per capire cosa "compra" un runtime gestito).
- [ ] 15.4 💡 🔭 **Dove eseguire Langfuse.** Due opzioni, ADR obbligatorio:
  - **Langfuse Cloud, regione UE**: nessuna operatività, dati in Irlanda, conformità documentata dal fornitore; da verificare con il legale come sub-responsabile del trattamento.
  - **Self-hosted su AWS UE**: `web` e `worker` su ECS, PostgreSQL su RDS, ClickHouse (self-managed su EC2/EKS oppure ClickHouse Cloud in regione UE), Redis su ElastiCache, S3 per blob ed export. Massimo controllo sui dati, ma sei tu a gestire aggiornamenti, backup e capacità.
  🧪 Fai almeno una volta il self-host **in locale** con Docker Compose per toccare con mano i componenti e capire cosa comporta gestirli.
- [ ] 15.5 **Identità.** Ruolo IAM del task/funzione con la policy minima del Modulo 13. Nessuna credenziale nell'immagine. Chiavi Langfuse da Secrets Manager.
- [ ] 15.6 **Server.** `axum` con health check (`/healthz`, `/readyz`), graceful shutdown (finisci le richieste in corso, cancella i task in background con criterio, **flush dell'exporter OTel**), timeout per richiesta, limiti di dimensione del body, rate limit per utente, CORS se serve.
- [ ] 15.7 **Stato.** SQLite non basta in produzione multi-istanza: PostgreSQL gestito (RDS) con `pgvector` per la KB. Migrazioni versionate (`sqlx migrate`).
- [ ] 15.8 💡 🔭 **Rollout graduale.** Per l'app: due versioni in parallelo (blue/green o canary sul 5% del traffico). Per i prompt: la versione attiva è la **label** `production` in Langfuse, quindi un rollback di prompt è uno spostamento di label, senza deploy; una label `canary` letta dal 5% delle richieste permette di provare un prompt nuovo su traffico reale, con gli score online (14.6) filtrati per versione del prompt a dire se promuoverlo. Tieni la cache dei prompt corta abbastanza (5–10 minuti) da rendere il rollback rapido.
- [ ] 15.9 💡 **Controlli di costo.** Budget e alert già attivi (3.8, 14.4); aggiungi un interruttore di emergenza (kill switch) che degrada il servizio (solo escalation e risposte fisse) se il costo orario supera una soglia.
- [ ] 15.10 🧪 **Test di carico.** Con `k6` o `oha`, 20–50 utenti simulati con messaggi sintetici. Osserva: throttling di Bedrock (le quote di 3.7), p95 per step in Langfuse, comportamento del router sotto stress, costo per minuto, ritardo di ingestione delle trace. Ritocca quote, timeout, circuit breaker e dimensione dei batch dell'exporter.
- [ ] 15.11 Pipeline di deploy: build su tag, push su ECR, deploy su ECS con approvazione manuale per produzione; la nightly del Modulo 12 deve essere verde per poter promuovere.
- [ ] 15.12 Runbook operativo: come fare rollback dell'app e di un prompt, come cambiare modello di emergenza (override nel router), come spegnere il tracing dei contenuti, chi chiamare.

**Fatto quando** l'app risponde in produzione con streaming, le trace arrivano in Langfuse con `environment = prod`, gli alert sono attivi, e hai fatto almeno un rollback di prompt spostando una label.

**Autoverifica.** Perché la versione attiva del prompt deve essere una label e non codice? Quali sono i costi nascosti del self-hosting di Langfuse?

---

## Modulo 16 · Approfondimenti opzionali

Da esplorare quando il core è solido. Ognuno è un piccolo esperimento con scheda nel Diario.

- [ ] 16.1 **Framework agentici a confronto.** Reimplementa l'agente del Modulo 7 con **Rig** (`rig-core` + `rig-bedrock`, il framework Rust più maturo, con tool tipizzati, RAG, MCP e attributi OpenTelemetry già inclusi). Confronta righe di codice, controllo sul loop, testabilità, osservabilità in Langfuse. Per capire cosa fa un framework "completo", guarda anche i concetti di LangChain 1.x (Python: `create_agent`, middleware, LangGraph) senza cambiare stack: il tuo harness **è** un agente con middleware scritti a mano. ADR "framework vs harness a mano".
- [ ] 16.2 **Multi-agente (orchestrator-workers).** Un orchestratore delega sotto-task a worker con modello economico e contesto isolato; poi sintetizza. In Langfuse la trace mostra l'albero completo. Confronta con la pipeline a singolo modello su un task di ricerca nella KB con più domande.
- [ ] 16.3 **Tool search e caricamento su richiesta.** Quando i tool sono decine, il loro elenco costa: tecniche per esporre solo i tool rilevanti per turno (indice locale, descrizioni brevi + schema completo su richiesta).
- [ ] 16.4 **Memoria come tool.** Dare al modello un tool `memory_read/write` su una directory o tabella, e lasciare che decida cosa ricordare; confronta con il profilo strutturato del Modulo 8.
- [ ] 16.5 **Langfuse più a fondo.** Prompt composti (un prompt che ne include altri, per riusare regole comuni tra `GENERATION` e `GUARDRAILS`); webhook sui cambi di prompt per far partire la nightly quando qualcuno sposta `production`; score a livello di **sessione** (qualità dell'intera conversazione, non del singolo messaggio); trace multimodali; export batch verso S3 e analisi con DuckDB.
- [ ] 16.6 **Bedrock AgentCore in profondità.** Runtime, Gateway (tool via MCP con autenticazione), Memory, Policy con Guardrails, Observability. Quanto del tuo codice sostituirebbe? Le sue trace possono convivere con Langfuse?
- [ ] 16.7 **Claude Platform on AWS.** Provare la stessa pipeline sull'accesso gestito da Anthropic in AWS (parità con l'API Anthropic: funzioni come compaction lato server, context editing, batch). Confronto onesto con Bedrock su funzionalità, prezzo, residenza dei dati.
- [ ] 16.8 **Batch inference.** Per `PROFILE_UPDATE` e per gli eval, il batch di Bedrock costa meno ma è asincrono: quando conviene?
- [ ] 16.9 **Fine-tuning e distillazione vs prompting.** Quando un classificatore fine-tuned piccolo batte un prompt su un modello grande (`SAFETY` è un candidato). Bedrock offre customizzazione per alcuni modelli; valuta costo e manutenzione. I dataset Langfuse annotati sono il punto di partenza per il training set.
- [ ] 16.10 **Migrazione a un nuovo modello.** Checklist: capacità (contract test), esperimenti completi confrontati in Langfuse con l'ultimo run buono, audit dei prompt per "incrostazioni" scritte per modelli vecchi, ricalibrazione di `effort`, costi, canary in produzione con score online, rollback pronto.
- [ ] 16.11 **Ottimizzazione dei costi come processo.** Ordine dei livelli: cache, igiene dei token in ingresso (contesto, tool result), igiene dell'output, batch, poi `effort`, poi cambio modello. Misura sempre il costo per **task completato**, che in Langfuse è il costo per trace, non per generation.
- [ ] 16.12 **Multimodale e voce.** Immagini in ingresso (Converse supporta blocchi immagine e documento), trascrizione e sintesi vocale come step della pipeline.

---

## Appendice A · Glossario (da compilare con parole tue)

| Termine | La tua definizione |
|---|---|
| Token | |
| Context window | |
| System prompt | |
| Tool use / function calling | |
| Harness | |
| Agente vs workflow | |
| Structured output | |
| Streaming / time-to-first-token | |
| Prompt caching / prefisso stabile | |
| Thinking adattivo / effort | |
| Inference profile (cross-region) | |
| Converse API | |
| Guardrail (LLM vs gestito) | |
| RAG / embedding / recall@k | |
| Trace / observation / generation | |
| Session / user (Langfuse) | |
| Score (numerico, categorico, booleano) | |
| Prompt version / label | |
| Dataset / dataset run / evaluator / annotation queue | |
| LLM-as-a-judge | |
| Intervallo di confidenza / rumore dell'eval | |
| Router / fallback / circuit breaker | |
| OpenTelemetry / OTLP / GenAI semconv | |
| MCP | |
| ADR | |

---

## Appendice B · Risorse

**Concetti e prompting (Anthropic)**
- Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
- Effective context engineering for AI agents: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Writing effective tools for agents: https://www.anthropic.com/engineering/writing-tools-for-agents
- Documentazione Claude (prompt engineering, structured outputs, tool use, note per Claude 5): https://platform.claude.com/docs

**Amazon Bedrock**
- Guida utente: https://docs.aws.amazon.com/bedrock/latest/userguide/
- Accesso ai modelli: https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html
- Converse API: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
- Structured output: https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
- Parametri Claude su Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
- Guardrails: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
- Intelligent Prompt Routing: https://aws.amazon.com/bedrock/intelligent-prompt-routing/
- AgentCore: https://aws.amazon.com/bedrock/agentcore/
- Prezzi: https://aws.amazon.com/bedrock/pricing/

**Langfuse**
- Documentazione: https://langfuse.com/docs
- Modello dati dell'osservabilità: https://langfuse.com/docs/observability/data-model
- Integrazione OpenTelemetry (endpoint, attributi): https://langfuse.com/integrations/native/opentelemetry
- Prompt management (concetti, versioni e label, caching, disponibilità garantita, collegamento alle trace): https://langfuse.com/docs/prompt-management/data-model
- Evaluation (concetti, dataset, esperimenti, evaluator, annotation queue): https://langfuse.com/docs/evaluation/overview
- API pubblica: https://langfuse.com/docs/api-and-data-platform/features/public-api
- Self-hosting (architettura v3, Docker Compose, Helm): https://langfuse.com/self-hosting
- Regioni dati e residenza in UE: https://langfuse.com/security/data-regions e https://langfuse.com/resources/engineering/langfuse-eu-data-residency-gdpr
- Changelog (le funzioni cambiano spesso): https://langfuse.com/changelog
- Crate Rust: `opentelemetry-langfuse`, `langfuse-ergonomic`, `langfuse-client-base` (organizzazione genai-rs su GitHub)

**Rust**
- Esempi Bedrock Runtime per Rust: https://docs.aws.amazon.com/sdk-for-rust/latest/dg/rust_bedrock-runtime_code_examples.html
- Test con l'SDK (mock e replay): https://docs.aws.amazon.com/sdk-for-rust/latest/dg/testing.html
- `aws-sdk-bedrockruntime` su docs.rs: https://docs.rs/aws-sdk-bedrockruntime
- SDK MCP ufficiale (`rmcp`): https://github.com/modelcontextprotocol/rust-sdk
- Rig (confronto opzionale): https://github.com/0xPlaygrounds/rig e crate `rig-bedrock`
- `schemars`, `insta`, `proptest`, `wiremock`, `aws-smithy-mocks`, `tracing-opentelemetry`, `opentelemetry-otlp` su crates.io

**Osservabilità e sicurezza**
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenTelemetry Collector: https://opentelemetry.io/docs/collector/
- Datadog LLM Observability (per la correlazione con l'infrastruttura): https://docs.datadoghq.com/llm_observability/
- OWASP Top 10 for LLM Applications: https://genai.owasp.org/

---

## Appendice C · Template ADR

File: `docs/adr/NNNN-titolo-breve.md`

```markdown
# NNNN · Titolo

- Data: AAAA-MM-GG
- Stato: proposto | accettato | superato da NNNN

## Contesto
Qual è il problema, quali vincoli (costo, latenza, privacy, tempo).

## Opzioni considerate
1. …  2. …  3. …

## Decisione
Cosa scegliamo e perché, con i numeri quando ci sono (link al dataset run in Langfuse e alla scheda esperimento).

## Conseguenze
Cosa diventa più facile, cosa più difficile, cosa va rivisto e quando.
```

---

## Appendice D · Scheda esperimento

Da compilare nel Diario per ogni 🧪.

```markdown
### EXP-NNN · Titolo · AAAA-MM-GG
- Domanda: …
- Variabile cambiata (una sola): …
- Configurazione: modello, prompt@versione, effort, dataset@versione, ripetizioni
- Dataset run in Langfuse: <link>
- Risultati: qualità (con IC), costo totale, p50/p95 latenza, errori infra
- Conclusione: …
- Prossimo passo: …
```

---

## Appendice E · Diario di bordo

Aggiungi in coda, una voce per sessione di lavoro. Poche righe: cosa hai fatto, cosa hai imparato, cosa non ti è chiaro.

```markdown
### AAAA-MM-GG
- Fatto: …
- Imparato: …
- Dubbi aperti: …
```
