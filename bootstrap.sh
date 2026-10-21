#!/usr/bin/env bash
set -euo pipefail

cat > settings.gradle.kts <<'F1'
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "FFAim"
include(":app")
F1

cat > build.gradle.kts <<'F2'
plugins {
    id("com.android.application") version "8.2.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
}
F2

mkdir -p app/src/main/java/com/bench/ffaim
mkdir -p app/src/main/res/xml
mkdir -p app/src/main/assets

cat > app/build.gradle.kts <<'F3'
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.bench.ffaim"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.bench.ffaim"
        minSdk = 26
        targetSdk = 34
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += "tflite" }
}
dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
}
F3

cat > app/src/main/AndroidManifest.xml <<'F4'
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW"/>
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PROJECTION"/>
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
    <application android:allowBackup="false"
        android:label="Screen Reader"
        android:theme="@android:style/Theme.Material.Light.NoActionBar">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        <service android:name=".ScreenCaptureService" android:exported="false"
            android:foregroundServiceType="mediaProjection"/>
        <service android:name=".OverlayService" android:exported="false"/>
        <service android:name=".AimAccessibilityService" android:exported="true"
            android:label="Screen Reader"
            android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE">
            <intent-filter>
                <action android:name="android.accessibilityservice.AccessibilityService"/>
            </intent-filter>
            <meta-data android:name="android.accessibilityservice"
                android:resource="@xml/aim_service_config"/>
        </service>
    </application>
</manifest>
F4

cat > app/src/main/res/xml/aim_service_config.xml <<'F5'
<?xml version="1.0" encoding="utf-8"?>
<accessibility-service xmlns:android="http://schemas.android.com/apk/res/android"
    android:accessibilityEventTypes="typeWindowStateChanged"
    android:accessibilityFeedbackType="feedbackGeneric"
    android:notificationTimeout="100"
    android:canPerformGestures="true"
    android:canRetrieveWindowContent="false"/>
F5

cat > app/src/main/java/com/bench/ffaim/Main.kt <<'F6'
package com.bench.ffaim

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.RectF
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.util.DisplayMetrics
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

object AimState {
    @Volatile var armed = false
    @Volatile var aimScale = 0.55f
    @Volatile var lastEngageMs = 0L
}

data class AimProfile(
    val fovPx: Float, val maxSwipePx: Float,
    val reactionMinMs: Long, val reactionMaxMs: Long,
    val cooldownMs: Long, val bodyBias: Float,
    val missChance: Float, val autoDisarmMs: Long,
    val minBodyHeightPx: Float, val minBodyRatio: Float, val maxBodyRatio: Float
)

object Profiles {
    val AWM_BODY = AimProfile(
        fovPx = 320f, maxSwipePx = 300f,
        reactionMinMs = 170L, reactionMaxMs = 290L,
        cooldownMs = 980L, bodyBias = -0.12f,
        missChance = 0.07f, autoDisarmMs = 8000L,
        minBodyHeightPx = 55f, minBodyRatio = 1.3f, maxBodyRatio = 4.0f
    )
}

data class Detection(val cx: Float, val cy: Float, val w: Float, val h: Float, val score: Float)

class BodyDetector(ctx: Context, asset: String = "yolov8n.tflite") {
    private val inputSize = 640
    private val interp: Interpreter
    private val inputBuf: ByteBuffer
    private val shape: IntArray
    private val fmt: Fmt
    private enum class Fmt { RAW, RAW_T, NMS }
    init {
        val opts = Interpreter.Options().apply {
            setNumThreads(4)
            try { addDelegate(NnApiDelegate()) } catch (_: Throwable) {}
        }
        val fd = ctx.assets.openFd(asset)
        val mm = FileInputStream(fd.fileDescriptor).use {
            it.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        interp = Interpreter(mm, opts)
        inputBuf = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).order(ByteOrder.nativeOrder())
        shape = interp.getOutputTensor(0).shape()
        fmt = when {
            shape.size == 3 && shape[2] == 6 && shape[1] in 100..3000 -> Fmt.NMS
            shape.size == 3 && shape[1] == 84 -> Fmt.RAW
            shape.size == 3 && shape[2] == 84 -> Fmt.RAW_T
            else -> throw IllegalStateException("shape ${shape.joinToString()}")
        }
    }
    fun detect(bmp: Bitmap, conf: Float = 0.45f, iouT: Float = 0.45f): List<Detection> {
        val sc = Bitmap.createScaledBitmap(bmp, inputSize, inputSize, true)
        val px = IntArray(inputSize * inputSize)
        sc.getPixels(px, 0, inputSize, 0, 0, inputSize, inputSize)
        inputBuf.rewind()
        for (p in px) {
            inputBuf.putFloat(((p shr 16) and 0xFF) / 255f)
            inputBuf.putFloat(((p shr 8) and 0xFF) / 255f)
            inputBuf.putFloat((p and 0xFF) / 255f)
        }
        if (sc !== bmp) sc.recycle()
        val out = ByteBuffer.allocateDirect(shape[1] * shape[2] * 4).order(ByteOrder.nativeOrder())
        interp.run(inputBuf, out)
        out.rewind()
        val cands = ArrayList<Detection>(16)
        when (fmt) {
            Fmt.NMS -> for (i in 0 until shape[1]) {
                val x1 = out.float; val y1 = out.float; val x2 = out.float; val y2 = out.float
                val s = out.float; val c = out.float
                if (c.toInt() != 0 || s < conf) continue
                cands.add(Detection((x1+x2)/2f, (y1+y2)/2f, x2-x1, y2-y1, s))
            }
            Fmt.RAW, Fmt.RAW_T -> {
                val d = shape[1]; val n = shape[2]
                fun v(r: Int, a: Int): Float = if (fmt == Fmt.RAW)
                    out.getFloat((r * n + a) * 4) else out.getFloat((a * d + r) * 4)
                for (a in 0 until n) {
                    val s = v(4, a)
                    if (s < conf) continue
                    cands.add(Detection(v(0,a), v(1,a), v(2,a), v(3,a), s))
                }
            }
        }
        cands.sortByDescending { it.score }
        val keep = ArrayList<Detection>(6)
        for (c in cands) {
            var ok = true
            for (k in keep) {
                val iw = max(0f, min(c.cx+c.w/2, k.cx+k.w/2) - max(c.cx-c.w/2, k.cx-k.w/2))
                val ih = max(0f, min(c.cy+c.h/2, k.cy+k.h/2) - max(c.cy-c.h/2, k.cy-k.h/2))
                val inter = iw * ih
                val u = c.w*c.h + k.w*k.h - inter
                if (u > 0f && inter/u > iouT) { ok = false; break }
            }
            if (ok) keep.add(c)
            if (keep.size >= 6) break
        }
        return keep
    }
    fun close() = interp.close()
}

class AimAccessibilityService : AccessibilityService() {
    companion object { @Volatile var instance: AimAccessibilityService? = null; private set }
    private val h = Handler(Looper.getMainLooper())
    override fun onServiceConnected() { instance = this }
    override fun onAccessibilityEvent(e: AccessibilityEvent?) {}
    override fun onInterrupt() {}
    fun swipeHumanized(x0: Float, y0: Float, x1: Float, y1: Float) {
        val dur = Random.nextLong(45L, 111L)
        val dx = x1 - x0; val dy = y1 - y0
        val dist = hypot(dx, dy).coerceAtLeast(1f)
        val px = -dy / dist; val py = dx / dist
        val path = Path().apply {
            moveTo(x0, y0)
            val segs = Random.nextInt(2, 5)
            var lx = x0; var ly = y0
            for (i in 1..segs) {
                val t = i.toFloat() / segs
                val bow = Random.nextFloat() * 12f - 6f
                val tx = Random.nextFloat() * 3f - 1.5f
                val ty = Random.nextFloat() * 3f - 1.5f
                val cx = lx + dx/segs + px*bow + tx
                val cy = ly + dy/segs + py*bow + ty
                val ex = x0 + dx*t + tx*0.5f
                val ey = y0 + dy*t + ty*0.5f
                quadraticBezierTo(cx, cy, ex, ey)
                lx = ex; ly = ey
            }
        }
        val stroke = GestureDescription.StrokeDescription(path, 0L, dur)
        dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, h)
    }
}

class AimController(
    private val a11y: AimAccessibilityService,
    private val screenW: Int, private val screenH: Int
) {
    private val profile = Profiles.AWM_BODY
    private val main = Handler(Looper.getMainLooper())
    private var lastSeen = 0L
    fun onDetections(dets: List<Detection>, crop: RectF, inferSize: Int) {
        if (!AimState.armed) return
        val now = SystemClock.uptimeMillis()
        if (now - lastSeen > profile.autoDisarmMs) { AimState.armed = false; return }
        if (now - AimState.lastEngageMs < profile.cooldownMs) return
        val cx = screenW / 2f; val cy = screenH / 2f
        val sx = crop.width() / inferSize; val sy = crop.height() / inferSize
        var best: Detection? = null; var bd = Float.MAX_VALUE
        for (d in dets) {
            val bh = d.h * sy
            if (bh < profile.minBodyHeightPx) continue
            val ratio = bh / (d.w * sx).coerceAtLeast(1f)
            if (ratio < profile.minBodyRatio || ratio > profile.maxBodyRatio) continue
            val bx = crop.left + d.cx * sx; val by = crop.top + d.cy * sy
            if (bx < screenW * 0.08f || bx > screenW * 0.92f) continue
            if (by < screenH * 0.18f || by > screenH * 0.88f) continue
            val dist = hypot(bx - cx, by - cy)
            if (dist <= profile.fovPx && dist < bd) { bd = dist; best = d }
        }
        val b = best ?: return
        lastSeen = now; AimState.lastEngageMs = now
        val bodyX = crop.left + b.cx * sx
        var bodyY = crop.top + (b.cy + b.h * profile.bodyBias) * sy
        if (Random.nextFloat() < profile.missChance) bodyY += b.h * sy * 0.25f
        val dx = ((bodyX - cx) * AimState.aimScale).coerceIn(-profile.maxSwipePx, profile.maxSwipePx)
        val dy = ((bodyY - cy) * AimState.aimScale).coerceIn(-profile.maxSwipePx, profile.maxSwipePx)
        if (abs(dx) < 3f && abs(dy) < 3f) return
        val delay = Random.nextLong(profile.reactionMinMs, profile.reactionMaxMs)
        main.postDelayed({
            val s0x = (screenW*0.72f + Random.nextFloat()*40f - 20f).coerceIn(80f, screenW-80f)
            val s0y = (screenH*0.58f + Random.nextFloat()*40f - 20f).coerceIn(80f, screenH-80f)
            val ex = (s0x + dx).coerceIn(20f, screenW-20f)
            val ey = (s0y + dy).coerceIn(20f, screenH-20f)
            a11y.swipeHumanized(s0x, s0y, ex, ey)
        }, delay)
    }
}

class ScreenCaptureService : Service() {
    companion object {
        const val ACTION_START = "start"
        const val EXTRA_CODE = "code"
        const val EXTRA_DATA = "data"
        @Volatile var running = false
        private const val INFER = 640
    }
    private var projection: MediaProjection? = null
    private var vd: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var worker: HandlerThread? = null
    private val inferExec = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var detector: BodyDetector? = null
    private var controller: AimController? = null
    override fun onBind(i: Intent?): IBinder? = null
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action != ACTION_START) { stopSelf(); return START_NOT_STICKY }
        if (running) return START_STICKY
        val code = intent.getIntExtra(EXTRA_CODE, 0)
        @Suppress("DEPRECATION")
        val data = intent.getParcelableExtra<Intent>(EXTRA_DATA) ?: run { stopSelf(); return START_NOT_STICKY }
        startForeground(42, notif())
        try {
            detector = BodyDetector(this)
        } catch (t: Throwable) { stopSelf(); return START_NOT_STICKY }
        val a11y = AimAccessibilityService.instance ?: run { stopSelf(); return START_NOT_STICKY }
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val m = DisplayMetrics()
        @Suppress("DEPRECATION") wm.defaultDisplay.getRealMetrics(m)
        val sw = m.widthPixels; val sh = m.heightPixels
        controller = AimController(a11y, sw, sh)
        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projection = mpm.getMediaProjection(code, data)
        projection?.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() { stopSelf() }
        }, Handler(mainLooper))
        reader = ImageReader.newInstance(sw, sh, PixelFormat.RGBA_8888, 3)
        worker = HandlerThread("cap").also { it.start() }
        val ch = Handler(worker!!.looper)
        reader!!.setOnImageAvailableListener({ r ->
            if (busy.getAndSet(true)) { r.acquireLatestImage()?.close(); return@setOnImageAvailableListener }
            val img = r.acquireLatestImage() ?: run { busy.set(false); return@setOnImageAvailableListener }
            try {
                val plane = img.planes[0]
                val rowPad = plane.rowStride - plane.pixelStride * img.width
                val paddedW = img.width + rowPad / plane.pixelStride
                val full = Bitmap.createBitmap(paddedW, img.height, Bitmap.Config.ARGB_8888)
                full.copyPixelsFromBuffer(plane.buffer)
                val side = minOf(sw, sh)
                val left = (sw - side) / 2; val top = (sh - side) / 2
                val sq = Bitmap.createBitmap(full, left, top, side, side)
                if (sq !== full) full.recycle()
                val scaled = Bitmap.createScaledBitmap(sq, INFER, INFER, true)
                if (scaled !== sq) sq.recycle()
                inferExec.execute {
                    try {
                        val dets = detector!!.detect(scaled)
                        scaled.recycle()
                        controller?.onDetections(
                            dets,
                            RectF(left.toFloat(), top.toFloat(), (left+side).toFloat(), (top+side).toFloat()),
                            INFER
                        )
                    } finally { busy.set(false) }
                }
            } catch (t: Throwable) { busy.set(false) } finally { img.close() }
        }, ch)
        vd = projection?.createVirtualDisplay(
            "cap", sw, sh, resources.displayMetrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader!!.surface, null, ch
        )
        running = true
        return START_STICKY
    }
    private fun notif(): Notification {
        val chId = "cap"
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && nm.getNotificationChannel(chId) == null)
            nm.createNotificationChannel(NotificationChannel(chId, "Screen Recorder", NotificationManager.IMPORTANCE_LOW))
        return Notification.Builder(this, chId)
            .setContentTitle("Screen Recorder").setContentText("Merekam layar")
            .setSmallIcon(android.R.drawable.ic_menu_camera).setOngoing(true).build()
    }
    override fun onDestroy() {
        running = false; AimState.armed = false
        reader?.close(); reader = null
        vd?.release(); vd = null
        projection?.stop(); projection = null
        worker?.quitSafely(); worker = null
        inferExec.shutdownNow()
        runCatching { detector?.close() }; detector = null
        super.onDestroy()
    }
}

class OverlayService : Service() {
    private var wm: WindowManager? = null
    private var view: FovView? = null
    private val h = Handler(Looper.getMainLooper())
    private val tick = object : Runnable {
        override fun run() { view?.armed = AimState.armed; view?.invalidate(); h.postDelayed(this, 200L) }
    }
    override fun onBind(i: Intent?): IBinder? = null
    override fun onCreate() {
        super.onCreate()
        wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val v = FovView(this)
        val p = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
            WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply { gravity = Gravity.TOP or Gravity.START }
        runCatching { wm?.addView(v, p) }
        view = v; h.post(tick)
    }
    override fun onDestroy() {
        h.removeCallbacks(tick)
        view?.let { runCatching { wm?.removeView(it) } }; view = null
        super.onDestroy()
    }
    private class FovView(ctx: Context) : View(ctx) {
        var armed = false
        private val idle = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2f; color = Color.argb(70, 0, 255, 180) }
        private val on = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f; color = Color.argb(200, 255, 60, 60) }
        private val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL; color = Color.argb(180, 255, 80, 80) }
        override fun onDraw(c: Canvas) {
            val cx = width / 2f; val cy = height / 2f
            c.drawCircle(cx, cy, 320f, if (armed) on else idle)
            c.drawCircle(cx, cy, 3f, dot)
        }
    }
}

class MainActivity : Activity() {
    private val REQ = 1001
    override fun onCreate(s: Bundle?) {
        super.onCreate(s)
        val ll = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(80, 120, 80, 80)
        }
        val status = TextView(this).apply { textSize = 14f; setPadding(0, 0, 0, 40) }
        fun refresh() {
            status.text = "A11y: ${AimAccessibilityService.instance != null}   Capture: ${ScreenCaptureService.running}   ARM: ${AimState.armed}"
        }
        refresh()
        val bOverlay = Button(this).apply {
            text = "1. Izin Overlay"
            setOnClickListener {
                startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")))
            }
        }
        val bA11y = Button(this).apply {
            text = "2. Aktifkan Screen Reader"
            setOnClickListener { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
        }
        val bCap = Button(this).apply {
            text = "3. Mulai Capture"
            setOnClickListener {
                val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
                @Suppress("DEPRECATION")
                startActivityForResult(mpm.createScreenCaptureIntent(), REQ)
            }
        }
        val bArm = Button(this).apply {
            text = "4. ARM / DISARM"
            setOnClickListener {
                if (!ScreenCaptureService.running) { Toast.makeText(this@MainActivity, "Capture dulu", Toast.LENGTH_SHORT).show(); return@setOnClickListener }
                if (AimAccessibilityService.instance == null) { Toast.makeText(this@MainActivity, "A11y dulu", Toast.LENGTH_SHORT).show(); return@setOnClickListener }
                startForegroundService(Intent(this@MainActivity, OverlayService::class.java))
                AimState.armed = !AimState.armed
                AimState.lastEngageMs = 0L
                refresh()
            }
        }
        val seek = SeekBar(this).apply {
            max = 100
            progress = ((AimState.aimScale - 0.20f) / 1.30f * 100).toInt().coerceIn(0, 100)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, f: Boolean) {
                    AimState.aimScale = 0.20f + p / 100f * 1.30f
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        ll.addView(status)
        ll.addView(bOverlay)
        ll.addView(bA11y)
        ll.addView(bCap)
        ll.addView(bArm)
        ll.addView(TextView(this).apply { text = "\nAim scale (kalibrasi)" })
        ll.addView(seek)
        setContentView(ll)
    }
    @Suppress("DEPRECATION")
    override fun onActivityResult(r: Int, res: Int, d: Intent?) {
        super.onActivityResult(r, res, d)
        if (r == REQ && res == RESULT_OK && d != null) {
            val i = Intent(this, ScreenCaptureService::class.java).apply {
                action = ScreenCaptureService.ACTION_START
                putExtra(ScreenCaptureService.EXTRA_CODE, res)
                putExtra(ScreenCaptureService.EXTRA_DATA, d)
            }
            startForegroundService(i)
        }
    }
}
F6

pip install --quiet ultralytics
yolo export model=yolov8n.pt format=tflite imgsz=640
MODEL=$(find . -name "*.tflite" -type f | head -1)
if [ -z "$MODEL" ]; then echo "ERROR: tflite tidak ditemukan"; exit 1; fi
cp "$MODEL" app/src/main/assets/yolov8n.tflite
ls -la app/src/main/assets/

gradle assembleDebug --no-daemon
ls -la app/build/outputs/apk/debug/
