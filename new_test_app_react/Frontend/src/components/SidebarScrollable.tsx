import * as ScrollArea from "@radix-ui/react-scroll-area";
import * as Tooltip from "@radix-ui/react-tooltip";
import { ChevronDown } from "lucide-react";
import { useRef, useState, useEffect } from "react";

export default function SidebarScrollable({ children }: { children: React.ReactNode }) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null); // track requestAnimationFrame
  const [showScrollDown, setShowScrollDown] = useState(false);
  const [hoverBottom, setHoverBottom] = useState(false);

  // Check scroll position
  const checkScroll = () => {
    const el = viewportRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 5;
    setShowScrollDown(el.scrollHeight > el.clientHeight && !atBottom);
  };

  useEffect(() => {
    checkScroll();
    const el = viewportRef.current;
    if (!el) return;
    el.addEventListener("scroll", checkScroll);
    window.addEventListener("resize", checkScroll);

    return () => {
      el.removeEventListener("scroll", checkScroll);
      window.removeEventListener("resize", checkScroll);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  const scrollToBottom = () => {
    const el = viewportRef.current;
    if (!el) return;

    if (animationRef.current) cancelAnimationFrame(animationRef.current);

    const start = el.scrollTop;
    const end = el.scrollHeight - el.clientHeight;
    const duration = 6000; // slow scroll
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.scrollTop = start + (end - start) * ease;

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);
  };

  // Stop auto-scroll when user interacts (click anywhere in viewport)
  const stopScroll = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };

  return (
    <div className="relative h-full w-full">
      <ScrollArea.Root className="h-full w-full">
        <ScrollArea.Viewport
          ref={viewportRef}
          className="h-full w-full"
          onClick={stopScroll} // stop on user interaction
        >
          {children}
        </ScrollArea.Viewport>

        <ScrollArea.Scrollbar
          orientation="vertical"
          className="w-2 bg-gray-200 dark:bg-gray-700"
        />
        <ScrollArea.Corner className="bg-gray-200 dark:bg-gray-700" />
      </ScrollArea.Root>

      {/* Floating Scroll-to-Bottom button */}
      {showScrollDown && (
        <div
          className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20"
          onMouseEnter={() => setHoverBottom(true)}
          onMouseLeave={() => setHoverBottom(false)}
        >
          <Tooltip.Provider delayDuration={200}>
            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <button
                  onClick={scrollToBottom}
                  className={[
                    // fade in/out
                    hoverBottom
                      ? "opacity-100 pointer-events-auto"
                      : "opacity-0 pointer-events-none",
                    "transition-opacity duration-200",

                    // circular button
                    "w-9 h-9 flex items-center justify-center rounded-full shadow-md",

                    // theme
                    "bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600",
                  ].join(" ")}
                >
                  <ChevronDown className="w-5 h-5 text-gray-800 dark:text-gray-200" />
                </button>
              </Tooltip.Trigger>

              <Tooltip.Content
                side="top"
                className="bg-gray-900 text-white text-xs rounded px-2 py-1 shadow-md"
              >
                Scroll to bottom
                <Tooltip.Arrow className="fill-gray-900" />
              </Tooltip.Content>
            </Tooltip.Root>
          </Tooltip.Provider>
        </div>
      )}

    </div>
  );
}
