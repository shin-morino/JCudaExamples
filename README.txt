本レポジトリには、2016/1/27開催の「MaxwellとJava、C#のためのCUDA」で使用したサンプルコードが収められています。

プレゼンテーションのPDFは、以下をご参照ください。
http://www.slideshare.net/NVIDIAJapan/maxwell-java-cuda-57590051

実行時には、ご使用のOSに応じ、platform_bin以下にある*.dll、*.soに対して、パスをとおしてください。

また、円周率のファイルは、分割してレポジトリに格納されています。
Windowsでは、concat.sh、Linuxでは、concat.shを実行して、ファイルを結合してください。


--
以下、今回利用したライブラリ、ソースについての説明です。
ライセンスは、それぞれのライブラリのライセンスに従います。

JCuda
JCudaのjar、dll、soについては、こちらで、githubからダウンロードしたv0.7.5のソースコードを、ビルドしています。
https://github.com/MysterionRise/mavenized-jcuda

JNA
以下、github中のバイナリを使用しています。Version 4.2.1です。
https://github.com/java-native-access/jna

BoyerMoore法のサンプルコードは、以下のURLの実装をベースに作成しています。
http://algs4.cs.princeton.edu/53substring/BoyerMoore.java.html

