from openai import OpenAI
import json

client = OpenAI(
    api_key='xxxx'
)


def generate_summarized_sentences(video_id, timestamps, sentences, fps, num_frames, num_variants=5):
    # Create the prompt for GPT-4
    prompt = (
        f"Generate {num_variants} unique, informative, and concise descriptions summarizing the key actions and events "
        f"in a video. The descriptions should cover the general context and sequence of activities without being overly detailed. "
        f"The video is about someone in the kitchen, performing various kitchen-related tasks. Make sure each variant is distinct, "
        f"using different phrasing or focusing on different aspects of the actions. Here's the sequence of actions from the video:\n\n"
    )

    for sentence in sentences:
        prompt += f"- {sentence}\n"

    prompt += (
        f"\nThe descriptions should summarize the video as a whole, capturing the essence of what's happening. "
        f"Return the result as {num_variants} distinct descriptions, without putting numbers at the beginings of the scentences, "
        f"just scentences, separated by '###', don't write these phrases at the beginning of the scenteces 'In this video' or 'This video is about' "
        f"just start with the description directly"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Extract the generated summaries from the response
        summarized_sentences = response.choices[0].message.content.split("###")

        return [
            {
                "video_id": video_id,
                "summarized_sentence": sentence.strip(),
                "fps": fps,
                "num_frames": num_frames
            }
            for sentence in summarized_sentences
        ]

    except Exception as e:
        print(f"Error generating summarized sentences for video {video_id}: {e}")
        return []


def process_video_data(input_json_file, output_json_file):
    with open(input_json_file, 'r') as infile:
        video_data = json.load(infile)

    records_count = len(video_data.items())
    counter = 1
    output_data = []
    for video_id, details in video_data.items():
        print(f"Processing {counter}/{records_count}")
        counter += 1
        timestamps = details['timestamps']
        sentences = details['sentences']
        fps = details['fps']
        num_frames = details['num_frames']

        # Generate summarized sentences using the OpenAI API
        summarized_data = generate_summarized_sentences(video_id, timestamps, sentences, fps, num_frames)
        output_data.extend(summarized_data)

    # Write the output to a new JSON file
    with open(output_json_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
        print(f"Output written to {output_json_file}")


if __name__ == "__main__":

    input_json_file = "train.json"
    output_json_file = "new_annotations/train.json"

    process_video_data(input_json_file, output_json_file)
