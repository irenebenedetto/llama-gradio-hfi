import gradio as gr
import requests
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s.%(funcName)s() - line: %(lineno)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--gradio_port", type=int, required=False, default=7860)
parser.add_argument("--server_port", type=int, required=False, default=8080)

args = parser.parse_args()
log.info(f"args: {args}")



examples = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    ["Gli ingredienti per una buona carbonara sono uova, pasta, ", "128", 80, 100, "Raw output"],

    ["La teoria della relativitÃ  di Einstein considera ", "512", 0, 100, "Post-process output"],

    # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
    [
        "Tweet: \"Odio la batteria del mio telefono\"\nSentiment: Negativo\n###\nTweet: \"Mi piace ascoltare la musica pop\"\nSentiment: Positivo\n###\nTweet: \"Questo Ã¨ il link all'articolo\"\nSentiment: Neutrale\n###\nTweet: \"Mia mamma ha fatto una pizza cattiva\"\nSentiment:",
        "8", 0, 100, "Post-process output"],

    [
        "Translate from English to Italian:\n- sea otter => lontra di mare\n- peppermint => menta piperita\n- laptop => computer portatile\n- sunflower =>",
        "8", 0, 100, "Post-process output"],

    [
        "# News: L'azzurra Sofia Goggia ha vinto la sua quarta coppa del mondo di discesa dopo quelle del 2018-2021 e 2022. La certezza matematica del successo le Ã¨ arrivata prima ancora di scendere in pista nella discesa di Kvitfijell e dopo che la sua unica potenziale rivale, la slovena Ilka Stuhec, si era piazzata momentaneamente terza, ottenendo cosi' comunque troppo pochi punti per poter superare Sofia.\n# Titolo: Sci: Sofia Goggia vince la sua quarta Coppa del mondo di discesa\n###\n# News: Fu il \"28 febbraio 2020\" il giorno cruciale che portÃ² l'Italia a non riuscire a contrastare la pandemia da Covid, malgrado i dati a disposizione, perchÃ© invece che alle zone rosse, come quella da applicare in Val Seriana, il Comitato tecnico scientifico si affidÃ² a \"misure proporzionali\" per combattere un \"virus che si propagava esponenzialmente\". Lo sostiene, nella sua consulenza per la Procura di Bergamo, il microbiologo Andrea Crisanti.\n# Titolo: Covid, Crisanti: 'Dal 28 febbraio scelte cruciali di improvvisazione. Il governo sapeva e non agÃ¬'\n### \n# News: \"Sono molto soddisfatta del bilancio di questa missione: entrambi i bilaterali andati molto bene, sinceramente anche oltre le mie aspettative\": cosÃ¬ la premier Giorgia Meloni in un punto stampa ad Abu Dhabi al termine della visita che ha visto l'India come prima tappa. \"Mi pare che ci sia ampia disponibilitÃ  da parte dello sceicco Bin Zayed di dare una mano nella piena volontÃ  di recuperare un rapporto di amicizia per gli interessi nazionali dell'UItalia\".  In questo viaggio \"abbiamo elevato i rapporti bilaterali e credo che con l'India ci sia un rapporto particolarmente importante. Da guida del G20 puÃ² giocare ruolo molto importante in diplomazia nelle crisi internazionali in atto\", ha detto ancora la premier.\n# Titolo: ",
        "128", 30, 90, "Post-process output"],

    [
        "#Â Commento: l'app Ã¨ meravigliosa, pratica e funzionale.\n# Commento: il sito web non funziona mai.\n# Commento: ore di attesa allo sportello per poi non risolvere nulla.\n#Â Commento: il personale Ã¨ molto gentile anche se spesso non Ã¨ in grado di rispondere prontamente alle mie domande.\nTemi trattati: l'app, gentilezza del personale, il sito web, i tempi di attesa.\n###Â \n# Commento: adoro la scelta dei colori della collezione autunno-inverno, molto originale.\n# Commento: il personale Ã¨ sempre molto scontroso, le mie richieste non vengono mai prese in carico.\n# Commento: il personale Ã¨ insufficiente per un negozio cosÃ¬ frequentato.\nTemi trattati:",
        "32", 0, 90, "Post-process output"],

    [
        "\n# Commento 1: l'app Ã¨ meravigliosa, pratica e funzionale. \n# Commento 2: il sito web non funziona mai. \n# Commento 3: ho ricevuto una pronta risposta dal personale allo sportello, molto preparato. \n# Commento 4: il personale Ã¨ molto gentile anche se spesso non Ã¨ in grado di rispondere prontamente alle mie domande. \nCommenti negativi: 2, 4 \n### \n# Commento 1: ci sono 3 bottoni che non funzionano. \n# Commento 2: ore di attesa sul sito e poi la procedura non funziona. \n# Commento 3: la lending page Ã¨ molto carina ed essenziale. \n# Commento 4: il personale Ã¨ molto gentile, e LLaMA di piÃ¹. \nCommenti negativi: 1, 2 \n### \n# Commento 1: adoro la scelta dei colori della collezione autunno-inverno, molto originale. \n# Commento 2 : il personale Ã¨ sempre molto scontroso, le mie richieste non vengono mai prese in carico. \n# Commento 3: il personale Ã¨ insufficiente per un negozio cosÃ¬ frequentato. \n# Commento 4: il negozio Ã¨ molto pulito e presenta una vasta gamma di prodotti. \nCommenti negativi:",
        "8", 0, 100, "Post-process output"],

    [
        "DOMANDA: Il concerto doveva iniziare alle 2 del pomeriggio ma il cantante Ã¨ arrivato 3 dopo, quindi il cantante Ã¨ arrivato alle?\nOPZIONI: (a) alle 5 del mattino (b) alle 6 del mattino (c) alle 5 del pomeriggio (d) alle sei del mattino.\nRISPOSTA:",
        "32", 50, 95, "Post-process output"],

    ["Il musicista Lucio Dalla Ã¨ nato il ", "32", 50, 95, "Post-process output"],

    [
        "Date le seguenti frasi:\n# Mia sorella adora giocare con il cane.\n# Mia sorella adora giocare con Bobby, il suo animale da compagnia.\nIl significato di queste frasi Ã¨ lo stesso in quanto animale da compagnia e cane sono sinonimi.\n###\nDate le seguenti frasi:\n# Ho comprato una paio di orecchini.\n# Utilizzo l'orologio da polso tutti i giorni.\nIl significato di queste frasi Ã¨ diverso in quanto trattano di due temi distinti (orecchini e utilizzo di orologio da polso).\n###\nDate le seguenti frasi:\n# Il televisore a LCD ha una risoluzione pazzesca a parer mio.\n# Adoro la qualitÃ  dell'immagine dei televisori LCD.\nIl significato di queste frasi Ã¨ ",
        "128", 0, 100, "Post-process output"]
]


def generate_text(text, max_gen_len=128, temperature=0.8, top_p=0.95, postprocess="Raw output"):

    d = {
        "text": text,
        "max_gen_len": max_gen_len,
        "temperature" : temperature,
        "top_p": top_p,
        "postprocess": postprocess
    }
    response = requests.post(f"http://0.0.0.0:{args.server_port}", data=d)
    return response.json()


with gr.Blocks() as demo:
    gr.Markdown(
        f"""
    # Demo for LLaMA ðŸ¦™
    """.strip())
    with gr.Row():
        text_input = gr.components.Textbox(lines=3, label="Input prompt")

    with gr.Row():
        max_gen_len = gr.Slider(minimum=1, maximum=1024, label="Max output length", interactive=True, value=128)
        postprocess = gr.Dropdown(["Post-process output", "Raw output"], label="Post-processing setting",
                                  interactive=True, value="Post-process output")

    with gr.Row():
        temperature = gr.Slider(minimum=0, maximum=100, label="Temperature sampling", interactive=True, value=80)
        top_p = gr.Slider(minimum=0, maximum=100, label="Top-p of distribution of probably of common tokens",
                          interactive=True, value=100)

    with gr.Row():
        output = gr.components.Textbox(lines=3, label="Answer")

    with gr.Row():
        submit = gr.Button("Submit", interactive=True)

        submit.click(generate_text, inputs=[text_input, max_gen_len, temperature, top_p, postprocess], outputs=output)

    with gr.Row():
        gr.Examples(
            examples=examples,
            inputs=[text_input, max_gen_len, temperature, top_p, postprocess],
            fn=generate_text,
            outputs=output,
            cache_examples=False,
        )


gr.close_all()

log.info("Starting demo...")
demo.launch(
    share=True,
    server_port=args.gradio_port,
    max_threads=1,
    auth=("admin", "cambiami")
)