import 'dart:convert';
import 'dart:math';
import 'package:image/image.dart';
import 'package:collection/collection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'dart:ui' as ui;

class Classifier {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;

  late List<int> _inputShape;
  late List<int> _outputShape;

  late TensorImage _inputImage;
  late TensorBuffer _outputBuffer;

  late TfLiteType _inputType;
  late TfLiteType _outputType;

  late var _probabilityProcessor;

  final String modelName = 'mnist.tflite';
  // final String _labelFileName = 'assets/labels.txt';
  // final int _labelLength = 1000;
  // final NormalizeOp preProcessNormalizeOp = NormalizeOp(114.495, 57.63);
  final NormalizeOp preProcessNormalizeOp = NormalizeOp(0, 255);
  final NormalizeOp postProcessNormalizeOp = NormalizeOp(0, 1);
  late List<String> labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

  Classifier({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();

    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }

    loadModel();
    // loadLabels();
  }

  // モデルデータの読み込み
  Future<void> loadModel() async {
    try {
      // 使用するモデルによって、入力・出力の形状・タイプが異なる
      // ここでは、mnistを実施するので、入力は[1, 28, 28]、出力は[1, 10]となる

      // この入力テンソルは、サイズが28x28ピクセルの2次元画像を表しています。
      // 最初の次元の値が1であることから、おそらく1つの画像を表していると推測されます。
      // 同様に、このテンソルは機械学習のコンテキストで使用される画像データの表現形式です。
      // 各ピクセルの値は、おそらく0から255までの範囲の整数で表されるグレースケールの画像であると想定されます。ピクセルの値が明示されていないため、具体的な画像の内容や特徴については言及できません。
      // このテンソルの次元[1, 28, 28]は、バッチサイズ、画像の高さ、画像の幅を表しています。
      // バッチサイズが1であることから、1つの画像を単独で処理していることがわかります。

      // この出力テンソルは、1つのデータポイントに対して10個の要素を持つベクトルを表しています。
      // 最初の次元の値が1であることから、おそらく1つのデータポイントに対する予測結果やクラス分類の確率分布を表していると推測されます。
      // このようなテンソルは、機械学習の分類や回帰のタスクにおいてよく使用されます。
      // 各要素は、異なるクラスまたはカテゴリに属する確率やスコアを表すことがあります。
      // データポイントの分類タスクの場合、最も高い値を持つ要素が予測されたクラスに対応していることが一般的です。
      // 具体的なデータセットやモデルの詳細に応じて、各要素がどのクラスを表しているのか、または何らかの数値的な予測結果を表しているのかを確認する必要があります。
      // また、テンソルの次元や形状だけではなく、モデルの出力の意味を正確に理解するためには、そのモデルやアルゴリズムの文脈も考慮する必要があります。

      interpreter =
          await Interpreter.fromAsset(modelName, options: _interpreterOptions);
      print('Interpreter Created Successfully');

      // テンソルの形状・タイプを取得
      var _tmp = interpreter.getInputTensor(0);
      _inputShape = interpreter.getInputTensor(0).shape;
      _outputShape = interpreter.getOutputTensor(0).shape;
      _inputType = interpreter.getInputTensor(0).type;
      _outputType = interpreter.getOutputTensor(0).type;

      print('Input Tensor Shape: $_tmp');
      print('Input Tensor Shape: $_inputShape');
      print('Input Tensor Type: $_inputType');
      print('Output Tensor Shape: $_outputShape');
      print('Output Tensor Type: $_outputType');

      // 出力バッファの作成
      _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);

      // 後処理の正規化操作を追加
      // ここでは、出力値を0〜1に正規化する
      _probabilityProcessor =
          TensorProcessorBuilder().add(postProcessNormalizeOp).build();
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  // ラベルデータの読み込み
  // Future<void> loadLabels() async {
  //   labels = await FileUtil.loadLabels(_labelFileName);
  //   if (labels.length == _labelLength) {
  //     print('Labels loaded successfully');
  //   } else {
  //     print('Unable to load labels');
  //   }
  // }

  // 画像の前処理
  TensorImage _preProcess() {
    // mnistを実施するので、28x28にクロップする
    int cropSize = 28;

    // 画像をリサイズして正規化する
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        // .add(preProcessNormalizeOp)
        .build()
        .process(_inputImage);
  }

  // 推論
  Category? predict(Image image) {
    print("---start predict---");
    // 画像の前処理
    // 入力画像をTensorImageに変換
    _inputImage = TensorImage(_inputType);
    // ImageをTensorImageにロード
    print('hoge');
    _inputImage.loadImage(image);
    // ============================
    // 画像をグレースケールで引数に渡しても、sizeが28*28*3になってしまう。なぜ。
    // ============================
    // 画像の前処理を実行
    print('hoge');
    _inputImage = _preProcess();
    print('hoge');

    // 入力テンソルの各次元を取得
    int ch = _inputShape[0];
    int h = _inputShape[1];
    int w = _inputShape[2];

    print("ch: $ch");
    print("h: $h");
    print("w: $w");

    // 入力画像をテンソル形式に変換
    // List inputImage = List.filled(ch * w * h, 0.0).reshape([1, ch, w, h]);
    // List inputImageList = _inputImage
    //     .getTensorBuffer()
    //     .getBuffer()
    //     .asFloat32List()
    //     .reshape([1, w, h, ch]);

    // ============================
    // 画像をグレースケールで引数に渡しても、sizeが28*28*3になってしまう。なぜ。
    // このsizeを[1,28,28]に変換してるけど、そもそもこれでいいのか？
    // 現状、sizeが28*28*3に合わして、inputImageListを作成して、その後[1,28,28]に変換している。
    // これがそもそも合ってないような気がする。
    // フォーマット自体はなんとか合ってるので、推論エラーにはならない。
    // ただ、推論結果がおかしい気がする、、、
    // ============================
    // 入力画像をテンソル形式に変換
    List inputImage = List.filled(ch * 28 * 28, 0.0).reshape([1, 28, 28]);
    List inputImageList = _inputImage
        .getTensorBuffer()
        .getBuffer()
        .asFloat32List()
        .reshape([1, 28, 28, 3]);

    // 入力画像の次元を変換: (1,224,224,3) => (1,3,224,224)
    // for (int c = 0; c < ch; c++) {
    //   for (int x = 0; x < w; x++) {
    //     for (int y = 0; y < h; y++) {
    //       inputImage[0][c][x][y] = inputImageList[0][x][y][c];
    //     }
    //   }
    // }

    // 入力画像の次元を変換: (1,28,28,3) => (1,28,28)
    for (int c = 0; c < ch; c++) {
      for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
          inputImage[0][x][y] = inputImageList[0][x][y][c];
        }
      }
    }

    // 推論を実行し、結果を_outputBufferに格納
    interpreter.run(inputImage, _outputBuffer.getBuffer());
    print(_outputBuffer.getBuffer());

    // ============================
    // 以下のブロックは未検証。推論結果の変換がおかしい確率も無きにしもあらず
    Map<String, double> labeledProb = TensorLabel.fromList(
            labels, _probabilityProcessor.process(_outputBuffer))
        .getMapWithFloatValue();

    labeledProb = softmax(labeledProb);
    final pred = getTopProbability(labeledProb);
    return Category(pred.key, pred.value);
    // ============================
    // return null;
  }

  Map<String, double> softmax(Map<String, double> labeledProb) {
    Map<String, double> ret;
    ret = labeledProb;
    var sum = labeledProb.values.reduce((a, b) => a + exp(b));
    labeledProb.forEach((key, value) {
      ret[key] = (exp(value)) / sum;
    });

    return ret;
  }

  MapEntry<String, double> getTopProbability(Map<String, double> labeledProb) {
    var pq = PriorityQueue<MapEntry<String, double>>(compare);
    pq.addAll(labeledProb.entries);
    return pq.first;
  }

  int compare(MapEntry<String, double> e1, MapEntry<String, double> e2) {
    if (e1.value > e2.value) {
      return -1;
    } else if (e1.value == e2.value) {
      return 0;
    } else {
      return 1;
    }
  }
}
















// import 'dart:math';
// import 'dart:typed_data';
// import 'package:image/image.dart';
// import 'package:collection/collection.dart';
// import 'package:tflite_flutter/tflite_flutter.dart';
// import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

// class Classifier {
//   late Interpreter interpreter;
//   late InterpreterOptions _interpreterOptions;

//   late List<int> _inputShape;
//   late List<int> _outputShape;

//   late TensorImage _inputImage;
//   late TensorBuffer _outputBuffer;

//   late TfLiteType _inputType;
//   late TfLiteType _outputType;

//   late var _probabilityProcessor;

//   final String modelName = 'mnist.tflite';
//   // final String _labelFileName = 'assets/labels.txt';
//   final int _labelLength = 1000;
//   final NormalizeOp preProcessNormalizeOp = NormalizeOp(114.495, 57.63);
//   final NormalizeOp postProcessNormalizeOp = NormalizeOp(0, 1);
//   late List<String> labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

//   Classifier({int? numThreads}) {
//     _interpreterOptions = InterpreterOptions();

//     if (numThreads != null) {
//       _interpreterOptions.threads = numThreads;
//     }

//     loadModel();
//     // loadLabels();
//   }

//   // モデルデータの読み込み
//   Future<void> loadModel() async {
//     try {
//       interpreter =
//           await Interpreter.fromAsset(modelName, options: _interpreterOptions);
//       print('Interpreter Created Successfully');

//       _inputShape = interpreter.getInputTensor(0).shape;
//       _outputShape = interpreter.getOutputTensor(0).shape;
//       _inputType = interpreter.getInputTensor(0).type;
//       _outputType = interpreter.getOutputTensor(0).type;

//       _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
//       _probabilityProcessor =
//           TensorProcessorBuilder().add(postProcessNormalizeOp).build();
//     } catch (e) {
//       print('Unable to create interpreter, Caught Exception: ${e.toString()}');
//     }
//   }

//   // ラベルデータの読み込み
//   // Future<void> loadLabels() async {
//   //   labels = await FileUtil.loadLabels(_labelFileName);
//   //   if (labels.length == _labelLength) {
//   //     print('Labels loaded successfully');
//   //   } else {
//   //     print('Unable to load labels');
//   //   }
//   // }

//   // 画像の前処理
//   TensorImage _preProcess() {
//     int cropSize = min(_inputImage.height, _inputImage.width);
//     return ImageProcessorBuilder()
//         .add(ResizeWithCropOrPadOp(cropSize, cropSize))
//         .add(ResizeOp(_inputShape[2], _inputShape[3], ResizeMethod.BILINEAR))
//         .add(preProcessNormalizeOp)
//         .build()
//         .process(_inputImage);
//   }

// // // MNIST画像の前処理
// //   TensorImage preprocessMNISTImage() {
// //     int cropSize = 28;

// //     return ImageProcessorBuilder()
// //         .add(ResizeWithCropOrPadOp(cropSize, cropSize))
// //         // .add(preProcessNormalizeOp)
// //         .build()
// //         .process(_inputImage);
// //   }

//   // 推論
//   Category predict(Image image) {
//     print("---start predict---");
//     // 画像の前処理
//     _inputImage = TensorImage(_inputType);
//     _inputImage.loadImage(image);
//     _inputImage = _preProcess();

//     int ch = _inputShape[1];
//     int w = _inputShape[2];
//     int h = _inputShape[3];

//     List inputImage = List.filled(ch * w * h, 0.0).reshape([1, ch, w, h]);
//     List inputImageList = _inputImage
//         .getTensorBuffer()
//         .getBuffer()
//         .asFloat32List()
//         .reshape([1, w, h, ch]);

//     // (1,224,224,3) => (1,3,224,224)
//     for (int c = 0; c < ch; c++) {
//       for (int x = 0; x < w; x++) {
//         for (int y = 0; y < h; y++) {
//           inputImage[0][c][x][y] = inputImageList[0][x][y][c];
//         }
//       }
//     }

//     // 推論実行
//     interpreter.run(inputImage, _outputBuffer.getBuffer());
//     Map<String, double> labeledProb = TensorLabel.fromList(
//             labels, _probabilityProcessor.process(_outputBuffer))
//         .getMapWithFloatValue();

//     labeledProb = softmax(labeledProb);
//     final pred = getTopProbability(labeledProb);
//     return Category(pred.key, pred.value);
//   }

//   Map<String, double> softmax(Map<String, double> labeledProb) {
//     Map<String, double> ret;
//     ret = labeledProb;
//     var sum = labeledProb.values.reduce((a, b) => a + exp(b));
//     labeledProb.forEach((key, value) {
//       ret[key] = (exp(value)) / sum;
//     });

//     return ret;
//   }

//   MapEntry<String, double> getTopProbability(Map<String, double> labeledProb) {
//     var pq = PriorityQueue<MapEntry<String, double>>(compare);
//     pq.addAll(labeledProb.entries);
//     return pq.first;
//   }

//   int compare(MapEntry<String, double> e1, MapEntry<String, double> e2) {
//     if (e1.value > e2.value) {
//       return -1;
//     } else if (e1.value == e2.value) {
//       return 0;
//     } else {
//       return 1;
//     }
//   }
// }
