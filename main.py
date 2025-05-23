# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import requests, os
import argparse
from PIL import Image


import gradio as gr
from together import Together
import textwrap


## FUNCTION 1: This Allows Us to Prompt the AI MODEL
# -------------------------------------------------
def prompt_llm(prompt, with_linebreak=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    # Calculate the number of tokens
    tokens = len(prompt.split())

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content

    if with_linebreak:
        # Wrap the output
        wrapped_output = textwrap.fill(output, width=50)

        return wrapped_output
    else:
        return output


## FUNCTION 2: This Allows Us to Generate Images
# -------------------------------------------------
def gen_image(prompt, width=256, height=256):
    # This function allows us to generate images from a prompt
    response = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell-Free",  # Using a supported model
        steps=2,
        n=1,
    )
    image_url = response.data[0].url
    image_filename = "image.png"

    # Download the image using requests instead of wget
    response = requests.get(image_url)
    with open(image_filename, "wb") as f:
        f.write(response.content)
    img = Image.open(image_filename)
    img = img.resize((height, width))

    return img


## Function 3: This Allows Us to Create a Chatbot
# -------------------------------------------------
def bot_response_function(user_message, chat_history):
    external_knowledge = """
    Drug Use for Grownups by Dr. Carl Hart, Chasing the Scream by Johann Hari, and This is Your Mind on Plants by Michael Pollan
    """

    chatbot_prompt = f"""
    You are a woke pharmacist who gives advice to patients on lifestyle choices, drugs and their side effects

    respond to this {user_message} following these instructions:

    ## Instructions:
    * be very concise
    * always start with Thank you for sharing
    * then encourage the user to have fun and make smart choices
    * Ground all your answers based on this book {external_knowledge} and make sure you cite the exact phrase from that book
    """
    print(f"DEBUG: User message: '{user_message}'")

    raw_llm_response = "Error: LLM call failed or returned unexpected data." # Default for error cases
    actual_response_for_user = "[LLM returned an empty or invalid response after processing]" # Default for post-processing

    try:
        api_response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[{"role": "user", "content": chatbot_prompt}],
        )
        raw_llm_response_content = api_response.choices[0].message.content
        if not isinstance(raw_llm_response_content, str):
            raw_llm_response_content = str(raw_llm_response_content)
            
        print(f"DEBUG: Raw LLM response (len={len(raw_llm_response_content)}): '{raw_llm_response_content}' (repr: {repr(raw_llm_response_content)})")

        # Attempt to extract content after </think>
        think_tag_end = "</think>"
        if think_tag_end in raw_llm_response_content:
            parts = raw_llm_response_content.split(think_tag_end, 1)
            if len(parts) > 1:
                actual_response_for_user = parts[1].strip()
            else: # Should not happen if tag is found, but as a safeguard
                actual_response_for_user = raw_llm_response_content.strip() 
        else:
            # If no <think> block, assume the whole response is for the user
            actual_response_for_user = raw_llm_response_content.strip()

        # If after all processing, the text is empty, use a placeholder
        if not actual_response_for_user:
            actual_response_for_user = "[LLM response was empty after stripping <think> block]"
        
        print(f"DEBUG: Extracted for user (len={len(actual_response_for_user)}): '{actual_response_for_user}'")

    except Exception as e:
        print(f"DEBUG: Error during LLM call or processing: {e}")
        actual_response_for_user = "Sorry, I encountered an error generating a response." # Keep this as a fallback

    # Assign to response_text for image prompt and history
    response_text = actual_response_for_user

    # Generate image based on the (cleaned) response
    image_prompt = f"A {response_text} in a pop art style"
    print(f"DEBUG: Image prompt: '{image_prompt}'")
    image = gen_image(image_prompt)

    chat_history.append((user_message, response_text))
    print(f"DEBUG: Updated chat history: {chat_history}")

    return "", chat_history, image


if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=int, default=1)
    parser.add_argument("-k", "--api_key", type=str, default=None)
    args = parser.parse_args()

    # Get Client for your LLMs
    client = Together(api_key=args.api_key)

    # run the script
    if args.option == 1:
        ### Task 1: YOUR CODE HERE - Write a prompt for the LLM to respond to the user
        prompt = "write a 3 line post about existentialism"

        # Get Response
        response = prompt_llm(prompt)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

    elif args.option == 2:
        ### Task 2: YOUR CODE HERE - Write a prompt for the LLM to generate an image
        prompt = "Create an image of Cthulu"

        print(f"\nCreating Image for your prompt: {prompt} ")
        img = gen_image(prompt=prompt, width=256, height=256)
        os.makedirs("results", exist_ok=True)
        img.save("results/image_option_2.png")
        print("\nImage saved to results/image_option_2.png\n")

    elif args.option == 3:
        ### Task 3: YOUR CODE HERE - Write a prompt for the LLM to generate text and an image
        text_prompt = "write a 3 line post about TikTok's effect on democracy for instagram"
        image_prompt = f"give me an image that represents this '{text_prompt}'"

        # Generate Text
        response = prompt_llm(text_prompt, with_linebreak=True)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

        # Generate Image
        print(f"\nCreating Image for your prompt: {image_prompt}... ")
        img = gen_image(prompt=image_prompt, width=256, height=256)
        img.save("results/image_option_3.png")
        print("\nImage saved to results/image_option_3.png\n")

    elif args.option == 4:
        # 4. Task 4: Create the chatbot interface (see bot_response_function for more details)
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("## 🤖 AI Chatbot")
            gr.Markdown("Enter your message below and let the chatbot respond!")

            chatbot = gr.Chatbot()
            image_output = gr.Image(label="Generated Image")
            user_input = gr.Textbox(
                placeholder="Type your message here...", label="Your Message"
            )
            send_button = gr.Button("Send")

            send_button.click(
                bot_response_function,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot, image_output],
            )
            user_input.submit(
                bot_response_function,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot, image_output],
            )

        app.launch()
    else:
        print("Invalid option")