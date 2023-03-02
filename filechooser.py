from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView


class FileChooserApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        fc = FileChooserListView(path='C:\\Users\\austi\\Documents')
        layout.add_widget(fc)
        button = Button(text='Choose')
        button.bind(on_press=self.choose_file)
        layout.add_widget(button)
        return layout

    def choose_file(self, button):
        # Get the selected file from the file chooser
        selected_file = self.root.children[0].selection[0]
        print(f'You selected {selected_file}')


if __name__ == '__main__':
    FileChooserApp().run()
