#!/usr/bin/env python3

# ... [previous imports and configurations remain unchanged]

def stream_speech(client, text_queue):
    full_text = ""
    buffer = ""
    process = None
    sentence_end_pattern = re.compile(r'(?<=[.!?])\s+')

    def send_chunk_to_tts(chunk):
        nonlocal process
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=chunk
            )
            if not process:
                process = subprocess.Popen(['ffplay', '-autoexit', '-nodisp', '-'], stdin=subprocess.PIPE)
            for audio_chunk in response.iter_bytes(chunk_size=4096):
                if audio_chunk:
                    process.stdin.write(audio_chunk)
                    process.stdin.flush()
        except Exception as e:
            logger.error(f"Error in TTS or audio playback: {str(e)}")

    try:
        while True:
            try:
                text_chunk = text_queue.get(timeout=0.1)  # Reduced timeout for faster processing
            except queue.Empty:
                if buffer:
                    send_chunk_to_tts(buffer)
                    buffer = ""
                break

            if text_chunk is None:
                break

            full_text += text_chunk
            buffer += text_chunk

            # Split the buffer into sentences
            sentences = sentence_end_pattern.split(buffer)

            # Process complete sentences if we have more than 100 words
            chunk_to_send = ""
            while len(sentences) > 1 and len(chunk_to_send.split()) < 100:
                chunk_to_send += sentences.pop(0) + " "

            # If we have at least 100 words and a complete sentence, send it
            if len(chunk_to_send.split()) >= 100:
                send_chunk_to_tts(chunk_to_send.strip())
                buffer = ''.join(sentences)
            else:
                buffer = chunk_to_send + ''.join(sentences)

        # Send any remaining text in the buffer
        if buffer:
            send_chunk_to_tts(buffer)

    except Exception as e:
        logger.error(f"Unexpected error in stream_speech: {str(e)}")
    finally:
        if process:
            process.stdin.close()
            process.wait()

# ... [rest of the script remains unchanged]

if __name__ == "__main__":
    main()
