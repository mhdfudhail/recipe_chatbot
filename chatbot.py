from openai import OpenAI


client = OpenAI(
  api_key="<<API-KEY>>"
)


def chat_with_gpt(prompt):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    #   store=True,
    messages=[
        {"role": "system", "content": "You are a helpful cooking assistant chatbot that specializes in recipes. "},
        {"role": "user", "content": prompt}
    ]
    )

    return response.choices[0].message.content.strip()

if __name__=="__main__":

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit","exit", "bye"]:
            break
        response = chat_with_gpt(user_input)
        print("Chatgpt: ", response)