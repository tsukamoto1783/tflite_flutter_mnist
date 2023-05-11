import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:tflite_sample_ver2/classifier.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Classification'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final picker = ImagePicker();
  String? _imagePath;
  File? _image;
  Image? _imageWidget;

  late Classifier _classifier;

  Category? category;

  @override
  void initState() {
    super.initState();
    _classifier = Classifier();
  }

  Future _getImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _imagePath = (pickedFile.path);
        _image = File(_imagePath!);
        _imageWidget = Image.file(_image!);

        _predict();
      });
    }
  }

  void _predict() async {
    img.Image imageInput = img.decodeImage(_image!.readAsBytesSync())!;
    img.Image grayscaleImage = img.grayscale(imageInput);
    var pred = _classifier.predict(grayscaleImage);
    setState(() {
      category = pred;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Center(
              child: _image == null
                  ? const Text("no image")
                  : Container(
                      child: _imageWidget,
                    ),
            ),
            const SizedBox(
              height: 36,
            ),
            // Text(
            //   category != null ? category!.label : "",
            //   style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
            // ),
            const SizedBox(
              height: 8,
            ),
            Text(
              category != null
                  ? 'Score: ${category!.score.toStringAsFixed(3)}'
                  : '',
              style: const TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _getImage,
        tooltip: 'Pick Image',
        child: const Icon(Icons.add_a_photo),
      ),
    );
  }
}
