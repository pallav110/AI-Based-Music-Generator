import edge_tts
import asyncio

async def main():
    # Your lyrics here
    text = """
    [Intro]
    
    [VERSE]
    pour up, terra no la e te de un monton
    or all of me sweep it
    around sum burn wats konvict.
    bear ah... come back up fire! waist bring en meri mas, en calle de sorat es noche
    vent dans un louco hai ha........ like he hit the volver of desires ilusin.

    [VERSE]
    for her contempt sillanp! trains and hurray the fire and the sergeant lies is consumption is burn by
    the land of desires hereditary lies tengas lo a celebration of discontent
    the blind lead us with the blind to arms, fear of death
    is screaming self obsessed burn the of lives for consumption feeding the away. self
    unholy wig the of with or in the land and my knees

    [CHORUS]
    and rest against the ground. to hate
    the price of the sky of a bottle or and the air
    the trees of god. hand of their rule and the winner. their power
    is the aesthetic of the ground. your and caution against their world

    [VERSE]
    into the passing of my brain, for all your hands he cripping
    the sheet blows the literature i'll run for a human please
    though the sun goes around you call me feel the next night my soul is like heaven were always resting
    for all the whiskey of the engine in a basement every feet will be leavin'
    you are sure that's the way they feel as we know the world and the hunted that i

    [CHORUS]
    feel all right this is the best way and live in the shadows of this world is
    that all there is and [music] is just the only way that i am in my soul [music] [applause] [music] it's
    the flash in the soul [music]
    [music] me oh yeah in the beginning there is a

    [BRIDGE]
    light in the [music] soul there is the
    [music] past [music] oh let us know the lord i set on you from a crowded
    into the city [music] [applause] [music] [applause]

    [CHORUS]
    oh [music] a hosana hosana in the
    highest hosana and worship you lord of the highest [music] hosana [music] we are here in the
    [music] world a [music]
    flash up hear my soul away and

    [CHORUS]
    sing our knees we're in my soul is fire my god hear us from [music] the flash the highest light and worship
    you need to make your name light sing your praise sing our praise to sing the
    praise of our glory cover sing our praise hosana bow bow [music] hosana bow up brea on the highest
    hosana and praise my soul the world of my heart lord lord lord lord i
    
    [Outro]
    """
    
    voice = "en-US-AriaNeural"  # You can change the voice here
    output_path = "tts_output.wav"

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    print(f"Saved TTS to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())







# PS C:\Users\pallav\Desktop\Python\music_generation_folder> python inference_main.py -m logs/44k -c logs/44k/config.json -n tts_output.wav -s Meek
# PS C:\Users\pallav\Desktop\Python\music_generation_folder> ffmpeg -i melodic_integrated_song.wav -i output.flac -filter_complex "[0:a]volume=0.8[a0];[1:a]volume=2.0[a1];[a0][a1]amix=inputs=2:duration=longest" full_song.wav                                                                                                                    
# >> 







                                                                                                                                                           