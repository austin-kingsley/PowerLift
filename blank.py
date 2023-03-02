import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.core.video import Video
from kivy.clock import Clock

class MyApp(App):
    def build(self):
        # Load the video file
        video = Video(source='sq1.mp4')

        # Create an image widget
        image = Image(source="foo.png")

        # Set the image widget's texture to the first frame of the video
        def set_image_texture(dt):
            texture = video.texture
            if texture:
                texture_size = list(texture.size)
                image.texture = texture
                image.texture_size = texture_size
                image.size = texture_size
                Clock.unschedule(set_image_texture)

        # Set the image texture when the video is loaded
        def on_load(video):
            video.play()
            Clock.schedule_interval(set_image_texture, 0.1)

        video.bind(on_load=on_load)

        return image

if __name__ == '__main__':
    MyApp().run()