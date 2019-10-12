package com.example.myapplication

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.view.Menu
import android.view.MenuItem

import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.content.Intent
import android.app.Activity
import android.database.Cursor
import android.net.Uri
import android.provider.MediaStore
import androidx.core.app.ActivityCompat
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.content_main.*
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import java.io.File
import java.io.FileOutputStream
import java.lang.Exception
import java.lang.RuntimeException

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {
    private var bm : Bitmap? = null
    private lateinit var model : Module
    private lateinit var imagePath : String
    private val _READSIG : Int = 114
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(toolbar)
        val f = File("$cacheDir/model.lite")
        if (!f.exists()) try {
            val modelStream = assets.open("model.lite")
            val size = modelStream.available()
            val buffer = ByteArray(size)
            modelStream.read(buffer)
            modelStream.close()
            val fos = FileOutputStream(f)
            fos.write(buffer)
            fos.close()
        } catch (e : Exception) {throw RuntimeException(e)}
        model = Module.load(f.path)
        fab.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, 1)
        }
    }
    @SuppressLint("Recycle")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 1 && resultCode == Activity.RESULT_OK && data != null) {
            val selectedImage : Uri = data.data!!
            val fileColumns: Array<out String> = arrayOf( MediaStore.Images.Media.DATA )
            val cursor : Cursor = contentResolver.query(selectedImage, fileColumns, null, null, null)!!
            cursor.moveToFirst()
            val columnIndex : Int = cursor.getColumnIndex(fileColumns[0])
            imagePath = cursor.getString(columnIndex)
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE), _READSIG)
            }
            bm = BitmapFactory.decodeFile(imagePath)
            imageView.setImageBitmap(bm)
            val inputBm = Bitmap.createScaledBitmap(bm!!, 224, 224, false)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(inputBm, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)
            val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray
            var maxScore = (-114514.1919).toFloat()
            var maxScoreIdx = -1
            for (i in scores.indices) {
                if (maxScore < scores[i]) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }
            val imageNetClasses = ImageNetClasses()
            val className = imageNetClasses.IMAGENET_CLASSES[maxScoreIdx]
            textView.text = className
            cursor.close()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int,
                                            permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == _READSIG) {
                // If request is cancelled, the result arrays are empty.
                if ((grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                }
                return
            }
    }
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }
}
